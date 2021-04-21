import pickle
from typing import Dict

import pandas as pd
from pandas import ExcelWriter
from tqdm import tqdm

def main():
    root_path = "/opt/ml/input/data/train_new"
    origin_path = f"{root_path}/train.tsv"

    # 원래 데이터 로드
    raw_origin: pd.DataFrame = pd.read_csv(
        origin_path,
        delimiter='\t',
        header=None
    )
    raw_origin.columns = ["source_code", "context", "sbj_entity",
                          "sbj_str", "sbj_end", "obj_entity", "obj_str", "obj_end", "label"]
    
    # -- 시작
    # 원래 데이터 형식 변환
    raw_converted = convert_orgin(raw_origin)

    # 부족분 업 샘플링
    # 5개도 안될 경우 Fold train set 안에 레이블이 없을 수도 있으므로
    upsample_to_limit(raw_converted)

    raw_converted.to_excel(f"{root_path}/train_new.xlsx", "combined_all", index=False, engine="xlsxwriter")


def upsample_to_limit(df: pd.DataFrame, limit: int):
    with open(f"/opt/ml/input/data/label_type.pkl", 'rb') as f:
        label_list: Dict = pickle.load(f)

    for label in label_list.keys():
        label_count = df[df["label"] == label].count()
        print(label_count)


def convert_orgin(source: pd.DataFrame):
    target = pd.DataFrame(columns=["source_code", "context", "sbj_entity", "obj_entity", "label"])

    counter_sbj = 0
    counter_obj = 0
    no_sbj = 0
    no_obj = 0

    for row in tqdm(source.iloc, total=len(source)):
        text: str = row["context"]
        sbj = row["sbj_entity"]
        sbj_start = row["sbj_str"]
        obj = row["obj_entity"]
        obj_start = row["obj_str"]
        
        ssp = sbj_start - 2
        osp = obj_start - 2
        # check_label_difference 함수에 의해 데이터의 위치 값의 오차는 2를 넘지 않음을 확인
        # 즉, label값에서 2만 빼준 위치에서 찾으면 제대로 된 위치값을 모두 얻을 수 있음
        fs_start = text.find(sbj, ssp if ssp >= 0 else 0)
        fs_end = fs_start + len(sbj)
        fo_start = text.find(obj, osp if osp >= 0 else 0)
        fo_end = fo_start + len(obj)


        if fs_start == -1:
            no_sbj += 1
        if fo_start == -1:
            no_obj += 1

        if fs_start != sbj_start:
            counter_sbj += 1
        if fo_start != obj_start:
            counter_obj += 1

        new_row = {
            "source_code": row["source_code"],
            "context": text,
            "sbj_entity": sbj,
            "obj_entity": obj,
            "label": row["label"],
        }
        if fs_start < fo_start:
            new_row["context"] = text[0: fs_start]
            new_row["context"] += "{{ sbj }}"
            new_row["context"] += text[fs_end:fo_start]
            new_row["context"] += "{{ obj }}"
            new_row["context"] += text[fo_end:]
        else:
            new_row["context"] = text[0: fo_start]
            new_row["context"] += "{{ obj }}"
            new_row["context"] += text[fo_end:fs_start]
            new_row["context"] += "{{ sbj }}"
            new_row["context"] += text[fs_end:]

        target = target.append(new_row, ignore_index=True)

    print("Position Label Check Result")
    print(f"sbj diff: {counter_sbj}")
    print(f"obj diff: {counter_obj}")
    print(f"fail to find sbj: {no_sbj}")
    print(f"fail to find obj: {no_obj}")    

    return target


if __name__ == "__main__":
    main()
