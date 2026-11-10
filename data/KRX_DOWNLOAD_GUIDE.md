# Rebuilding the KRX foreign-investor flow data

KRX's terms of use do not allow the firm-level investor-type trading records to be redistributed, so they are not in this repository. Anyone can download the same records free of charge from the KRX Data Marketplace and rebuild the target series with the code here.

## 1. Firms

`krx_firm_list.csv` lists the 150 firms: the number used in the code, the six-digit stock code, and the Korean and English names. It is the same list as Supplementary Table S1.

## 2. Download from the KRX Data Marketplace

1. Sign in at <https://data.krx.co.kr>. A free account is required for the statistics menus.
2. Open **통계 → 기본 통계 → 주식 → 거래실적 → 투자자별 거래실적(개별종목)** (Trading by investor type, individual stock).
3. For each firm, set the following:
   - **종목 (stock):** the stock code from `krx_firm_list.csv`
   - **조회구분 (view):** 일별추이 (daily series)
   - **거래량/거래대금:** 거래대금 (trading value, KRW)
   - **매수/매도/순매수:** 순매수 (net buy)
   - **조회기간 (period):** 2015-02-27 to 2026-06-26. A single query covers at most about two years, so download these six periods:

     | Period |
     |---|
     | 2015-02-27 ~ 2017-02-24 |
     | 2017-02-27 ~ 2019-02-27 |
     | 2019-02-28 ~ 2021-02-26 |
     | 2021-03-02 ~ 2023-03-02 |
     | 2023-03-03 ~ 2025-02-28 |
     | 2025-03-04 ~ 2026-06-26 |

4. Download each result as CSV. The file should have these columns:

   ```
   일자, 기관 합계, 기타법인, 개인, 외국인 합계, 전체
   ```

   Only `외국인 합계` (foreign investors, net-buy value) is used. Firms listed after 2015 simply have shorter histories.

## 3. Folder layout

Save the files under `data/krx_raw/` (this folder is git-ignored). Use one group folder per block of firm numbers and one folder per firm, named `<no>_<Korean name>`:

```
data/krx_raw/
  1~30/
    1_삼성전자/
      삼성전자_2015.02.27~2017.02.24.csv
      ...
      삼성전자_2025.03.04~2026.06.26.csv
    2_SK하이닉스/
    ...
  31~60/
  61~100/
  101~150/
    ...
    130_RFHIC/
    131_큐에스아이/
    ...
```

Each firm folder must contain only that firm's files, with no subfolders. The script reads every CSV under a firm folder recursively.

## 4. Build the series

```bash
python code/02_krx/combine_krx_150.py
```

The script writes the daily wide panel and the column `외국인_순매수금액_총합`: the foreign net-buy value summed over the firms with data on each day. This sum is `foreign_net_buy` in the model input panel (see `input_schema.md`). The target for day *t* is the sign of this sum on day *t + 1*.
