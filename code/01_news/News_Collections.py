import json
import re
import time
import random
import os
import threading

from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import feedparser
import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from tqdm import tqdm


# ============================================================
# 0. 저장 경로
# ============================================================

OUTPUT_DIR = Path(
    r".\1_News\Foreign_News"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 중요:
# 기존 파일명을 그대로 사용
#
# 기존에 수집한 결과 + checkpoint를 이어서 사용함
# ------------------------------------------------------------

RAW_FILE = (
    OUTPUT_DIR
    / "Korea_Semiconductor_ForeignInvestor_GoogleNews_RAW.csv"
)

FINAL_FILE = (
    OUTPUT_DIR
    / "Korea_Semiconductor_ForeignInvestor_Reuters_AP_Bloomberg.csv"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "GoogleNews_RSS_checkpoint.json"
)

ERROR_LOG = (
    OUTPUT_DIR
    / "GoogleNews_RSS_errors.txt"
)


# ============================================================
# 1. 연구 기간
# ============================================================

START_DATE = datetime(
    2015,
    2,
    18,
)

END_DATE = datetime(
    2026,
    6,
    26,
)


# ============================================================
# 2. 검색 범위 설정
#
# 검색 범위는 기존 코드 그대로 유지
#
# 약 66,000개 task
# ============================================================

WINDOW_DAYS = 30


# ============================================================
# 3. 속도 설정
# ============================================================

# ------------------------------------------------------------
# 동시에 실행할 HTTP 요청 수
#
# 우선 8 권장
#
# 429 오류가 거의 없으면:
# MAX_WORKERS = 12
#
# 429가 많이 발생하면:
# MAX_WORKERS = 4 ~ 6
# ------------------------------------------------------------

MAX_WORKERS = 8


# ------------------------------------------------------------
# 기존 30초 대신 8초
#
# 응답하지 않는 요청 하나 때문에
# 전체가 오래 기다리지 않도록 함
# ------------------------------------------------------------

REQUEST_TIMEOUT = 8


# ------------------------------------------------------------
# 실패 재시도
#
# 기존 3회 -> 2회
# ------------------------------------------------------------

MAX_RETRIES = 2


# ------------------------------------------------------------
# retry 사이 기본 대기
# ------------------------------------------------------------

RETRY_BASE_SLEEP = 1.5


# ------------------------------------------------------------
# 병렬 요청 자체에 아주 작은 랜덤 delay
#
# 완전히 동시에 Google에 요청하는 것을 완화
# ------------------------------------------------------------

REQUEST_JITTER_MIN = 0.02
REQUEST_JITTER_MAX = 0.15


# ------------------------------------------------------------
# 몇 개 task마다 CSV / checkpoint 저장할지
#
# 기존:
# task 하나마다 저장
#
# 수정:
# 100개마다 한 번 저장
# ------------------------------------------------------------

SAVE_EVERY_TASKS = 100


# ------------------------------------------------------------
# None = 전체 실행
#
# 테스트하려면:
# MAX_TASKS = 200
# ------------------------------------------------------------

MAX_TASKS = None


# ============================================================
# 4. 대상 외신
# ============================================================

SOURCES = {

    "Reuters":
        "reuters.com",

    "AP":
        "apnews.com",

    "Bloomberg":
        "bloomberg.com",
}


# ============================================================
# 5. 검색어
#
# 기존 159개 검색어 유지
# ============================================================

SEARCH_QUERIES = {


    # ========================================================
    # A. 한국 증시 외국인 투자자
    # ========================================================

    "foreign_investor_korea_01":
        '"foreign investors" "South Korea" stocks',

    "foreign_investor_korea_02":
        '"foreign investors" Korean stocks',

    "foreign_investor_korea_03":
        '"foreign investors" KOSPI',

    "foreign_investor_korea_04":
        '"foreign investors" KOSDAQ',

    "foreign_investor_korea_05":
        '"foreign institutional investors" Korea',

    "foreign_investor_korea_06":
        '"overseas investors" Korean stocks',

    "foreign_investor_korea_07":
        '"offshore investors" Korea stocks',

    "foreign_investor_korea_08":
        '"foreign ownership" Korean stocks',


    # ========================================================
    # B. 외국인 순매수 / 순매도 / 자금 흐름
    # ========================================================

    "foreign_flow_01":
        '"foreign buying" Korea stocks',

    "foreign_flow_02":
        '"foreign selling" Korea stocks',

    "foreign_flow_03":
        '"foreign net buying" Korea',

    "foreign_flow_04":
        '"foreign net selling" Korea',

    "foreign_flow_05":
        '"foreign fund flows" Korea stocks',

    "foreign_flow_06":
        '"foreign capital inflows" Korean stocks',

    "foreign_flow_07":
        '"foreign capital outflows" Korean stocks',

    "foreign_flow_08":
        '"foreign investors" "net buyers" Korea',

    "foreign_flow_09":
        '"foreign investors" "net sellers" Korea',


    # ========================================================
    # C. 외국인 투자자 + 반도체
    # ========================================================

    "foreign_semiconductor_01":
        '"foreign investors" Korea semiconductor',

    "foreign_semiconductor_02":
        '"foreign investors" Korean semiconductor',

    "foreign_semiconductor_03":
        '"foreign investors" Korea chip',

    "foreign_semiconductor_04":
        '"foreign investors" Korean chip stocks',

    "foreign_semiconductor_05":
        '"foreign buying" Korea semiconductor',

    "foreign_semiconductor_06":
        '"foreign selling" Korea semiconductor',

    "foreign_semiconductor_07":
        '"foreign net buying" semiconductor Korea',

    "foreign_semiconductor_08":
        '"foreign net selling" semiconductor Korea',

    "foreign_semiconductor_09":
        '"foreign ownership" semiconductor Korea',

    "foreign_semiconductor_10":
        '"foreign institutional investors" semiconductor Korea',


    # ========================================================
    # D. 삼성전자 + 외국인 투자자
    # ========================================================

    "samsung_foreign_01":
        '"Samsung Electronics" "foreign investors"',

    "samsung_foreign_02":
        '"Samsung Electronics" "foreign buying"',

    "samsung_foreign_03":
        '"Samsung Electronics" "foreign selling"',

    "samsung_foreign_04":
        '"Samsung Electronics" "foreign ownership"',

    "samsung_foreign_05":
        '"Samsung Electronics" "foreign institutional investors"',

    "samsung_foreign_06":
        '"Samsung Electronics" foreign funds',


    # ========================================================
    # E. SK hynix + 외국인 투자자
    # ========================================================

    "hynix_foreign_01":
        '"SK hynix" "foreign investors"',

    "hynix_foreign_02":
        '"SK Hynix" "foreign buying"',

    "hynix_foreign_03":
        '"SK Hynix" "foreign selling"',

    "hynix_foreign_04":
        '"SK Hynix" "foreign ownership"',

    "hynix_foreign_05":
        '"SK Hynix" "foreign institutional investors"',

    "hynix_foreign_06":
        '"SK Hynix" foreign funds',


    # ========================================================
    # F. 한국 반도체 산업 일반
    # ========================================================

    "semiconductor_01":
        '"South Korea" semiconductor',

    "semiconductor_02":
        '"South Korea" semiconductors',

    "semiconductor_03":
        '"Korean semiconductor industry"',

    "semiconductor_04":
        '"South Korea" chip industry',

    "semiconductor_05":
        '"Korean chip industry"',

    "semiconductor_06":
        '"South Korea" chipmaker',

    "semiconductor_07":
        '"South Korea" chipmakers',


    # ========================================================
    # G. 메모리 반도체
    # ========================================================

    "memory_01":
        '"South Korea" memory chip',

    "memory_02":
        '"Korea" memory semiconductor',

    "memory_03":
        '"South Korea" DRAM',

    "memory_04":
        '"South Korea" NAND',

    "memory_05":
        '"South Korea" HBM',

    "memory_06":
        '"Korea" high bandwidth memory',

    "memory_07":
        '"Korean" memory chips',


    # ========================================================
    # H. AI / HBM
    # ========================================================

    "ai_chip_01":
        '"South Korea" AI chip',

    "ai_chip_02":
        '"South Korea" AI semiconductor',

    "ai_chip_03":
        '"Korea" HBM semiconductor',

    "ai_chip_04":
        '"Samsung" HBM',

    "ai_chip_05":
        '"SK Hynix" HBM',

    "ai_chip_06":
        '"SK hynix" high bandwidth memory',

    "ai_chip_07":
        '"Samsung Electronics" high bandwidth memory',


    # ========================================================
    # I. Foundry
    # ========================================================

    "foundry_01":
        '"South Korea" foundry',

    "foundry_02":
        '"Korea" semiconductor foundry',

    "foundry_03":
        '"Samsung" foundry',

    "foundry_04":
        '"Samsung Electronics" foundry',

    "foundry_05":
        '"Korean chip foundry"',


    # ========================================================
    # J. Fab / Wafer / Manufacturing
    # ========================================================

    "manufacturing_01":
        '"South Korea" semiconductor fab',

    "manufacturing_02":
        '"South Korea" chip fab',

    "manufacturing_03":
        '"South Korea" wafer',

    "manufacturing_04":
        '"Korea" wafer fabrication',

    "manufacturing_05":
        '"Korea" semiconductor manufacturing',

    "manufacturing_06":
        '"Korea" chip manufacturing',


    # ========================================================
    # K. 반도체 장비
    # ========================================================

    "equipment_01":
        '"South Korea" semiconductor equipment',

    "equipment_02":
        '"Korea" chip equipment',

    "equipment_03":
        '"Korean semiconductor equipment"',

    "equipment_04":
        '"South Korea" lithography semiconductor',

    "equipment_05":
        '"Korea" chipmaking equipment',


    # ========================================================
    # L. 반도체 소재
    # ========================================================

    "materials_01":
        '"South Korea" semiconductor materials',

    "materials_02":
        '"Korea" chip materials',

    "materials_03":
        '"Korea" silicon wafer',

    "materials_04":
        '"Korea" photoresist semiconductor',

    "materials_05":
        '"Korean semiconductor materials"',


    # ========================================================
    # M. Advanced packaging
    # ========================================================

    "packaging_01":
        '"South Korea" advanced packaging semiconductor',

    "packaging_02":
        '"Korea" semiconductor packaging',

    "packaging_03":
        '"Korea" chip packaging',

    "packaging_04":
        '"Samsung" semiconductor packaging',

    "packaging_05":
        '"SK Hynix" packaging',


    # ========================================================
    # N. 반도체 수출
    # ========================================================

    "exports_01":
        '"South Korea" semiconductor exports',

    "exports_02":
        '"South Korea" chip exports',

    "exports_03":
        '"Korea" memory chip exports',

    "exports_04":
        '"Korea" semiconductor export growth',

    "exports_05":
        '"Korea" semiconductor export decline',

    "exports_06":
        '"Korea" chip export growth',

    "exports_07":
        '"Korea" chip export decline',


    # ========================================================
    # O. 수요 / 가격 / 업황
    # ========================================================

    "market_01":
        '"Korea" semiconductor demand',

    "market_02":
        '"Korea" chip demand',

    "market_03":
        '"Korea" memory chip prices',

    "market_04":
        '"Korea" DRAM prices',

    "market_05":
        '"Korea" NAND prices',

    "market_06":
        '"Korea" semiconductor downturn',

    "market_07":
        '"Korea" chip downturn',

    "market_08":
        '"Korea" semiconductor recovery',

    "market_09":
        '"Korea" chip recovery',

    "market_10":
        '"Korea" semiconductor cycle',


    # ========================================================
    # P. 공급망
    # ========================================================

    "supply_chain_01":
        '"South Korea" semiconductor supply chain',

    "supply_chain_02":
        '"South Korea" chip supply chain',

    "supply_chain_03":
        '"Korea" semiconductor shortage',

    "supply_chain_04":
        '"Korea" chip shortage',

    "supply_chain_05":
        '"Korea" semiconductor inventory',

    "supply_chain_06":
        '"Korea" chip inventory',


    # ========================================================
    # Q. 미국 정책 / CHIPS Act
    # ========================================================

    "policy_us_01":
        '"South Korea" CHIPS Act',

    "policy_us_02":
        '"Samsung" CHIPS Act',

    "policy_us_03":
        '"SK Hynix" CHIPS Act',

    "policy_us_04":
        '"Korea" US semiconductor policy',

    "policy_us_05":
        '"Korea" US chip policy',


    # ========================================================
    # R. 수출 통제
    # ========================================================

    "export_control_01":
        '"South Korea" semiconductor export controls',

    "export_control_02":
        '"South Korea" chip export controls',

    "export_control_03":
        '"Samsung" export controls China chips',

    "export_control_04":
        '"SK Hynix" export controls China',

    "export_control_05":
        '"Korea" US China chip restrictions',

    "export_control_06":
        '"Korea" semiconductor restrictions China',


    # ========================================================
    # S. 중국 관련
    # ========================================================

    "china_01":
        '"South Korea" China semiconductor',

    "china_02":
        '"Korea" China chip industry',

    "china_03":
        '"Samsung" China semiconductor',

    "china_04":
        '"SK Hynix" China semiconductor',

    "china_05":
        '"Samsung" China chip plant',

    "china_06":
        '"SK Hynix" China chip plant',


    # ========================================================
    # T. 일본 관련
    # ========================================================

    "japan_01":
        '"South Korea" Japan semiconductor',

    "japan_02":
        '"South Korea" Japan chip materials',

    "japan_03":
        '"Korea" Japan semiconductor exports',

    "japan_04":
        '"Korea" Japan chip restrictions',


    # ========================================================
    # U. 대만 / TSMC 경쟁
    # ========================================================

    "taiwan_01":
        '"South Korea" TSMC',

    "taiwan_02":
        '"Samsung" TSMC foundry',

    "taiwan_03":
        '"Samsung Electronics" TSMC',

    "taiwan_04":
        '"Korea" Taiwan semiconductor competition',


    # ========================================================
    # V. NVIDIA / AI 수요
    # ========================================================

    "nvidia_01":
        '"SK Hynix" Nvidia',

    "nvidia_02":
        '"Samsung Electronics" Nvidia HBM',

    "nvidia_03":
        '"Samsung" Nvidia semiconductor',

    "nvidia_04":
        '"Korea" Nvidia HBM',


    # ========================================================
    # W. 주요 국내 반도체 기업
    # ========================================================

    "company_01":
        '"Samsung Electronics" semiconductor',

    "company_02":
        '"Samsung Electronics" chip',

    "company_03":
        '"SK hynix" semiconductor',

    "company_04":
        '"SK Hynix" chip',

    "company_05":
        '"DB HiTek"',

    "company_06":
        '"Hanmi Semiconductor"',

    "company_07":
        '"LX Semicon"',

    "company_08":
        '"SFA Semicon"',

    "company_09":
        '"Soulbrain" semiconductor',

    "company_10":
        '"Wonik IPS" semiconductor',


    # ========================================================
    # X. 한국 정부 반도체 정책
    # ========================================================

    "korea_policy_01":
        '"South Korea" semiconductor policy',

    "korea_policy_02":
        '"South Korea" chip policy',

    "korea_policy_03":
        '"South Korea" semiconductor subsidy',

    "korea_policy_04":
        '"South Korea" chip subsidy',

    "korea_policy_05":
        '"South Korea" semiconductor tax credit',

    "korea_policy_06":
        '"Korea" semiconductor investment plan',


    # ========================================================
    # Y. 반도체 투자
    # ========================================================

    "investment_01":
        '"South Korea" semiconductor investment',

    "investment_02":
        '"Samsung" semiconductor investment',

    "investment_03":
        '"SK Hynix" semiconductor investment',

    "investment_04":
        '"Samsung" chip investment',

    "investment_05":
        '"SK Hynix" chip investment',
}


# ============================================================
# 6. Thread별 HTTP Session
#
# requests.Session을 여러 thread가 동시에 공유하지 않고
# thread마다 독립 Session을 사용
# ============================================================

_thread_local = threading.local()


def get_session():

    if not hasattr(
        _thread_local,
        "session"
    ):

        session = requests.Session()

        session.headers.update({

            "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
        })


        adapter = HTTPAdapter(
            pool_connections=4,
            pool_maxsize=4,
            max_retries=0,
        )

        session.mount(
            "https://",
            adapter
        )

        session.mount(
            "http://",
            adapter
        )

        _thread_local.session = (
            session
        )

    return (
        _thread_local.session
    )


# ============================================================
# 7. 기간 Window 생성
# ============================================================

def generate_windows(
    start_date,
    end_date,
    days
):

    current = start_date

    final_exclusive = (
        end_date
        + timedelta(days=1)
    )

    while current < final_exclusive:

        window_end = min(

            current
            + timedelta(days=days),

            final_exclusive
        )

        yield (
            current,
            window_end
        )

        current = (
            window_end
        )


# ============================================================
# 8. Google News RSS URL 생성
# ============================================================

def make_google_news_rss_url(
    keyword,
    domain,
    start_date,
    end_date,
):

    start_str = (
        start_date.strftime(
            "%Y-%m-%d"
        )
    )

    end_str = (
        end_date.strftime(
            "%Y-%m-%d"
        )
    )


    query = (

        f'{keyword} '

        f'site:{domain} '

        f'after:{start_str} '

        f'before:{end_str}'
    )


    encoded = (
        quote_plus(
            query
        )
    )


    url = (

        "https://news.google.com/rss/search?"

        f"q={encoded}"

        "&hl=en-US"

        "&gl=US"

        "&ceid=US:en"
    )


    return (
        url,
        query
    )


# ============================================================
# 9. HTML / 문자열 정리
# ============================================================

def clean_text(value):

    if value is None:

        return ""


    text = str(value)


    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )


    replacements = {

        "&nbsp;": " ",

        "&amp;": "&",

        "&quot;": '"',

        "&#39;": "'",

        "&lt;": "<",

        "&gt;": ">",
    }


    for old, new in (
        replacements.items()
    ):

        text = (
            text.replace(
                old,
                new
            )
        )


    text = re.sub(
        r"\s+",
        " ",
        text,
    )


    return (
        text.strip()
    )


# ============================================================
# 10. 날짜 변환
# ============================================================

def parse_date(value):

    if not value:

        return ""


    try:

        dt = pd.to_datetime(
            value,
            errors="coerce",
            utc=True,
        )


        if pd.isna(dt):

            return ""


        return (
            dt.strftime(
                "%Y-%m-%d"
            )
        )


    except Exception:

        return ""


# ============================================================
# 11. Publisher
# ============================================================

def get_publisher(
    entry,
    fallback,
):

    try:

        source = (
            entry.get(
                "source"
            )
        )


        if source:

            if isinstance(
                source,
                dict
            ):

                title = (
                    source.get(
                        "title"
                    )
                )

            else:

                title = getattr(
                    source,
                    "title",
                    None,
                )


            if title:

                return (
                    str(title)
                    .strip()
                )


    except Exception:

        pass


    return fallback


# ============================================================
# 12. 제목 정리
# ============================================================

def clean_title(
    title,
    publisher,
):

    title = (
        clean_text(
            title
        )
    )


    if publisher:

        title = re.sub(

            r"\s+-\s+"
            +
            re.escape(
                publisher
            )
            +
            r"\s*$",

            "",

            title,

            flags=re.IGNORECASE,
        )


    return (
        title.strip()
    )


# ============================================================
# 13. Checkpoint
# ============================================================

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return set()


    try:

        with open(

            CHECKPOINT_FILE,

            "r",

            encoding="utf-8",

        ) as f:

            data = (
                json.load(f)
            )


        return set(
            data.get(
                "completed",
                []
            )
        )


    except Exception:

        return set()


# ============================================================
# 14. Checkpoint 안전 저장
#
# 임시파일 저장 후 replace
# ============================================================

def save_checkpoint(
    completed
):

    temp_file = (
        CHECKPOINT_FILE
        .with_suffix(
            ".tmp"
        )
    )


    with open(

        temp_file,

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            {
                "completed":
                    sorted(
                        completed
                    )
            },

            f,

            ensure_ascii=False,

            indent=2,
        )


    os.replace(
        temp_file,
        CHECKPOINT_FILE,
    )


# ============================================================
# 15. 오류 로그
# ============================================================

def append_error_logs(
    errors
):

    if not errors:

        return


    with open(

        ERROR_LOG,

        "a",

        encoding="utf-8",

    ) as f:

        for item in errors:

            f.write(
                f"task="
                f"{item['task_id']}\n"
            )

            f.write(
                f"type="
                f"{item['type']}\n"
            )

            f.write(
                f"error="
                f"{item['error']}\n"
            )

            f.write(
                "-" * 80
                + "\n"
            )


# ============================================================
# 16. RSS 요청
# ============================================================

def request_feed(
    rss_url
):

    session = (
        get_session()
    )


    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            # 여러 thread가 완전히 동시에 요청하지 않도록
            time.sleep(
                random.uniform(
                    REQUEST_JITTER_MIN,
                    REQUEST_JITTER_MAX,
                )
            )


            response = (
                session.get(

                    rss_url,

                    timeout=(
                        REQUEST_TIMEOUT
                    ),
                )
            )


            # Rate Limit
            if (
                response.status_code
                ==
                429
            ):

                raise (
                    requests.HTTPError(
                        "HTTP 429 Too Many Requests"
                    )
                )


            response.raise_for_status()


            feed = (
                feedparser.parse(
                    response.content
                )
            )


            return feed


        except Exception as e:

            last_error = e


            if (
                attempt
                <
                MAX_RETRIES
            ):

                # exponential backoff + jitter
                wait = (
                    RETRY_BASE_SLEEP
                    *
                    (2 ** (attempt - 1))
                    +
                    random.uniform(
                        0.2,
                        0.8,
                    )
                )


                time.sleep(
                    wait
                )


    raise last_error


# ============================================================
# 17. 기사 파싱
# ============================================================

def parse_entry(
    entry,
    query_group,
    keyword,
    source_name,
    source_domain,
    query,
):


    publisher = (
        get_publisher(

            entry,

            source_name,
        )
    )


    title = (
        clean_title(

            entry.get(
                "title",
                ""
            ),

            publisher,
        )
    )


    date = (
        parse_date(

            entry.get(
                "published",
                ""
            )
        )
    )


    summary = (
        clean_text(

            entry.get(
                "summary",
                ""
            )
        )
    )


    google_url = (
        entry.get(
            "link",
            ""
        )
    )


    return {

        "date":
            date,

        "title":
            title,

        "publisher":
            publisher,

        "target_source":
            source_name,

        "target_domain":
            source_domain,

        "google_news_url":
            google_url,

        "rss_summary":
            summary,

        "query_group":
            query_group,

        "search_keyword":
            keyword,

        "search_query":
            query,
    }


# ============================================================
# 18. Task 생성
# ============================================================

def build_tasks(
    completed
):

    windows = list(

        generate_windows(

            START_DATE,

            END_DATE,

            WINDOW_DAYS,
        )
    )


    all_tasks = []


    for source_name, domain in (
        SOURCES.items()
    ):


        for (
            query_group,
            keyword
        ) in SEARCH_QUERIES.items():


            for (
                window_start,
                window_end
            ) in windows:


                task_id = (

                    f"{source_name}|"

                    f"{query_group}|"

                    f"{window_start:%Y-%m-%d}|"

                    f"{window_end:%Y-%m-%d}"
                )


                if (
                    task_id
                    in
                    completed
                ):

                    continue


                all_tasks.append({

                    "task_id":
                        task_id,

                    "source_name":
                        source_name,

                    "domain":
                        domain,

                    "query_group":
                        query_group,

                    "keyword":
                        keyword,

                    "window_start":
                        window_start,

                    "window_end":
                        window_end,
                })


    if MAX_TASKS is not None:

        all_tasks = (
            all_tasks[
                :MAX_TASKS
            ]
        )


    return (
        all_tasks,
        windows,
    )


# ============================================================
# 19. Task 1건 실행
#
# Thread에서 실행되는 함수
#
# CSV에는 여기서 쓰지 않음.
# 메인 Thread에서만 저장.
# ============================================================

def process_task(task):

    task_id = (
        task[
            "task_id"
        ]
    )


    try:

        rss_url, query = (
            make_google_news_rss_url(

                task[
                    "keyword"
                ],

                task[
                    "domain"
                ],

                task[
                    "window_start"
                ],

                task[
                    "window_end"
                ],
            )
        )


        feed = (
            request_feed(
                rss_url
            )
        )


        rows = []


        for entry in (
            feed.entries
        ):


            row = (
                parse_entry(

                    entry,

                    task[
                        "query_group"
                    ],

                    task[
                        "keyword"
                    ],

                    task[
                        "source_name"
                    ],

                    task[
                        "domain"
                    ],

                    query,
                )
            )


            rows.append(
                row
            )


        return {

            "success":
                True,

            "task_id":
                task_id,

            "rows":
                rows,

            "error":
                None,
        }


    except Exception as e:


        return {

            "success":
                False,

            "task_id":
                task_id,

            "rows":
                [],

            "error": {

                "task_id":
                    task_id,

                "type":
                    type(e).__name__,

                "error":
                    repr(e),
            },
        }


# ============================================================
# 20. Buffer 저장
# ============================================================

def flush_buffers(
    rows_buffer,
    completed_buffer,
    error_buffer,
    completed,
):


    # --------------------------------------------------------
    # 1. 기사 결과 먼저 저장
    # --------------------------------------------------------

    if rows_buffer:

        df = (
            pd.DataFrame(
                rows_buffer
            )
        )


        df.to_csv(

            RAW_FILE,

            mode="a",

            header=(
                not RAW_FILE.exists()
            ),

            index=False,

            encoding="utf-8-sig",
        )


    # --------------------------------------------------------
    # 2. 정상 완료 Task를 checkpoint에 반영
    # --------------------------------------------------------

    if completed_buffer:

        completed.update(
            completed_buffer
        )


        save_checkpoint(
            completed
        )


    # --------------------------------------------------------
    # 3. 오류 로그
    # --------------------------------------------------------

    if error_buffer:

        append_error_logs(
            error_buffer
        )


# ============================================================
# 21. 전체 병렬 수집
# ============================================================

def collect_news():

    completed = (
        load_checkpoint()
    )


    tasks, windows = (
        build_tasks(
            completed
        )
    )


    total_possible_tasks = (

        len(SOURCES)

        *
        len(SEARCH_QUERIES)

        *
        len(windows)
    )


    print("=" * 80)

    print(
        "Reuters / AP / Bloomberg"
    )

    print(
        "한국 반도체 + 외국인 투자자 뉴스 RSS 병렬 수집"
    )

    print("=" * 80)


    print(
        f"시작일        : "
        f"{START_DATE:%Y-%m-%d}"
    )


    print(
        f"종료일        : "
        f"{END_DATE:%Y-%m-%d}"
    )


    print(
        f"언론사        : "
        f"{len(SOURCES)}개"
    )


    print(
        f"검색어        : "
        f"{len(SEARCH_QUERIES):,}개"
    )


    print(
        f"검색 Window   : "
        f"{WINDOW_DAYS}일"
    )


    print(
        f"전체 작업     : "
        f"{total_possible_tasks:,}건"
    )


    print(
        f"기존 완료     : "
        f"{len(completed):,}건"
    )


    print(
        f"이번 실행 대상: "
        f"{len(tasks):,}건"
    )


    print(
        f"병렬 Worker   : "
        f"{MAX_WORKERS}개"
    )


    print(
        f"Timeout       : "
        f"{REQUEST_TIMEOUT}초"
    )


    print(
        f"저장 단위     : "
        f"{SAVE_EVERY_TASKS} Tasks"
    )


    print("=" * 80)


    if not tasks:

        print(
            "\n이미 모든 작업이 완료되었습니다."
        )

        return


    rows_buffer = []

    completed_buffer = []

    error_buffer = []


    success_count = 0
    error_count = 0
    article_count = 0
    tasks_since_save = 0


    start_time = (
        time.time()
    )


    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:


        future_to_task = {

            executor.submit(
                process_task,
                task
            ):
                task

            for task in tasks
        }


        progress = tqdm(

            total=len(tasks),

            desc="Google News RSS",

            unit="task",
        )


        try:

            for future in as_completed(
                future_to_task
            ):


                try:

                    result = (
                        future.result()
                    )


                except Exception as e:

                    task = (
                        future_to_task[
                            future
                        ]
                    )


                    error_count += 1


                    error_buffer.append({

                        "task_id":
                            task[
                                "task_id"
                            ],

                        "type":
                            type(e).__name__,

                        "error":
                            repr(e),
                    })


                    tasks_since_save += 1

                    progress.update(1)

                    continue


                # ------------------------------------------------
                # 정상
                # ------------------------------------------------

                if result[
                    "success"
                ]:


                    success_count += 1


                    completed_buffer.append(

                        result[
                            "task_id"
                        ]
                    )


                    if result[
                        "rows"
                    ]:


                        rows_buffer.extend(

                            result[
                                "rows"
                            ]
                        )


                        article_count += len(

                            result[
                                "rows"
                            ]
                        )


                # ------------------------------------------------
                # 오류
                # ------------------------------------------------

                else:

                    error_count += 1


                    if result[
                        "error"
                    ]:

                        error_buffer.append(

                            result[
                                "error"
                            ]
                        )


                tasks_since_save += 1


                # ------------------------------------------------
                # 100 Tasks 단위 저장
                # ------------------------------------------------

                if (
                    tasks_since_save
                    >=
                    SAVE_EVERY_TASKS
                ):


                    flush_buffers(

                        rows_buffer,

                        completed_buffer,

                        error_buffer,

                        completed,
                    )


                    rows_buffer.clear()

                    completed_buffer.clear()

                    error_buffer.clear()


                    tasks_since_save = 0


                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                elapsed = (
                    time.time()
                    -
                    start_time
                )


                speed = (

                    (
                        success_count
                        +
                        error_count
                    )
                    /
                    elapsed

                    if elapsed > 0

                    else 0
                )


                progress.set_postfix({

                    "ok":
                        success_count,

                    "err":
                        error_count,

                    "articles":
                        article_count,

                    "task/s":
                        f"{speed:.2f}",
                })


                progress.update(1)


        except KeyboardInterrupt:

            print(
                "\n\nCtrl+C 감지."
            )

            print(
                "현재 Buffer 저장 후 종료합니다..."
            )


            # 아직 실행 중인 future cancel 시도
            for future in (
                future_to_task
            ):

                future.cancel()


        finally:

            # ------------------------------------------------
            # 마지막 buffer 저장
            # ------------------------------------------------

            flush_buffers(

                rows_buffer,

                completed_buffer,

                error_buffer,

                completed,
            )


            progress.close()


    elapsed = (
        time.time()
        -
        start_time
    )


    print("\n" + "=" * 80)

    print(
        "RSS 수집 세션 종료"
    )

    print("=" * 80)


    print(
        f"성공 Task : "
        f"{success_count:,}"
    )


    print(
        f"오류 Task : "
        f"{error_count:,}"
    )


    print(
        f"수집 기사 : "
        f"{article_count:,}"
    )


    print(
        f"누적 완료 : "
        f"{len(completed):,}"
        f"/{total_possible_tasks:,}"
    )


    if elapsed > 0:

        print(
            f"평균 속도 : "
            f"{(success_count + error_count) / elapsed:.2f} "
            f"tasks/sec"
        )


    print("=" * 80)


# ============================================================
# 22. 최종 데이터 정리
# ============================================================

def finalize():

    if not RAW_FILE.exists():

        print(
            "\nRAW 파일이 없습니다."
        )

        return


    print(
        "\n최종 데이터 정리 중..."
    )


    df = pd.read_csv(

        RAW_FILE,

        encoding="utf-8-sig",

        low_memory=False,
    )


    print(
        f"RAW 검색 결과: "
        f"{len(df):,}건"
    )


    # ========================================================
    # 날짜
    # ========================================================

    df[
        "date"
    ] = pd.to_datetime(

        df[
            "date"
        ],

        errors="coerce",
    )


    df = df[

        (
            df["date"]
            >=
            START_DATE
        )

        &

        (
            df["date"]
            <=
            END_DATE
        )

    ].copy()


    # ========================================================
    # 제목
    # ========================================================

    df[
        "title"
    ] = (

        df[
            "title"
        ]

        .fillna("")

        .astype(str)

        .str.strip()
    )


    df = df[

        df[
            "title"
        ]
        !=
        ""

    ].copy()


    # ========================================================
    # 제목 normalization
    # ========================================================

    df[
        "title_normalized"
    ] = (

        df[
            "title"
        ]

        .str.lower()

        .str.replace(

            r"[^a-z0-9가-힣]+",

            " ",

            regex=True,
        )

        .str.replace(

            r"\s+",

            " ",

            regex=True,
        )

        .str.strip()
    )


    # ========================================================
    # 중복 제거
    #
    # 날짜 + 언론사 + 제목
    # ========================================================

    before = (
        len(df)
    )


    df = (
        df.drop_duplicates(

            subset=[

                "date",

                "target_source",

                "title_normalized",
            ],

            keep="first",
        )
        .copy()
    )


    print(
        f"중복 제거: "
        f"{before - len(df):,}건"
    )


    # ========================================================
    # 날짜 순
    # ========================================================

    df = (

        df.sort_values(

            [

                "date",

                "target_source",

                "title",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # article_id
    # ========================================================

    df.insert(

        0,

        "article_id",

        range(
            1,
            len(df) + 1,
        )
    )


    df[
        "date"
    ] = (

        df[
            "date"
        ]

        .dt.strftime(
            "%Y-%m-%d"
        )
    )


    # ========================================================
    # 최종 컬럼
    # ========================================================

    final_columns = [

        "article_id",

        "date",

        "title",

        "publisher",

        "target_source",

        "target_domain",

        "rss_summary",

        "google_news_url",

        "query_group",

        "search_keyword",
    ]


    df = (
        df[
            final_columns
        ]
    )


    df.to_csv(

        FINAL_FILE,

        index=False,

        encoding="utf-8-sig",
    )


    print("\n" + "=" * 80)

    print(
        "최종 데이터 저장 완료"
    )

    print("=" * 80)


    print(
        f"최종 기사 수: "
        f"{len(df):,}건"
    )


    if not df.empty:


        print(
            "\n언론사별 기사:"
        )


        print(

            df[
                "target_source"
            ]

            .value_counts()

            .to_string()
        )


        print(
            "\n연도별 기사:"
        )


        years = (

            pd.to_datetime(
                df[
                    "date"
                ]
            )

            .dt.year
        )


        print(

            years

            .value_counts()

            .sort_index()

            .to_string()
        )


    print(
        f"\nRAW 파일:\n"
        f"{RAW_FILE}"
    )


    print(
        f"\n최종 파일:\n"
        f"{FINAL_FILE}"
    )


    print("=" * 80)


# ============================================================
# 23. 실행
# ============================================================

if __name__ == "__main__":

    collect_news()

    finalize()