import os
import pickle
import re
from typing import Dict

import pandas as pd
from pandas import ExcelWriter
from tqdm import tqdm


def main():
    origin_path = "./data/train/train2.tsv"
    additional_path = "./results/추가 데이터.xlsx"

    # 원래 데이터 로드
    raw_origin: pd.DataFrame = pd.read_csv(
        origin_path,
        delimiter='\t',
        header=None
    )
    raw_origin.columns = ["source_code", "context", "sbj_entity",
                          "sbj_str", "sbj_end", "obj_entity", "obj_str", "obj_end", "label"]

    # 추가 데이터(kor-re-gold) 로드
    raw_new = pd.read_excel(additional_path, "new_raw")

    # 레이블 로드. 엑셀에서 읽을 수 있도록 변환하기 위한 코드. 필요없음.
    with open(f"./data/label_type.pkl", 'rb') as f:
        label_type: Dict = pickle.load(f)
        label_type["blind"] = 100
    label_table = pd.DataFrame(list(label_type.keys()))
    
    # -- 시작
    # 원래 데이터 형식 변환
    check_label_difference(raw_origin)  # sbj, obj 위치 라벨 오류 확인
    raw_converted = convert_orgin(raw_origin)

    # new data 형식 변환
    raw_new = convert_new_data(raw_new)

    # 특문자 수정
    refine_special_letter(raw_converted)
    refine_special_letter(raw_new)

    # 데이터 병합
    combined_raw = pd.concat([raw_converted, raw_new])
    combined_raw.reset_index(drop=True)

    if not os.path.isdir("./data/train_new"):
        os.mkdir("./data/train_new")
    with ExcelWriter("./data/train_new/train_new.xlsx", engine="xlsxwriter") as writer:
        # 기본 excel 엔진에 문제가 있는지 경고가 뜸
        combined_raw.to_excel(writer, "combined_all", index=False)
        raw_converted.to_excel(writer, "origin", index=False)
        raw_origin.to_excel(writer, "origin_raw", index=False)
        raw_new.to_excel(writer, "added", index=False)
        raw_new.to_excel(writer, "added_raw", index=False)
        label_table.to_excel(writer, "labels_list", index=False)

        writer.save()


def convert_orgin(source: pd.DataFrame):
    target = pd.DataFrame(columns=["source_code", "context", "sbj_entity", "obj_entity", "label"])

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

    return target

        

def convert_new_data(source: pd.DataFrame, no_append_for_none: bool = True):
    target = pd.DataFrame(columns=["source_code", "context", "sbj_entity", "obj_entity", "label"])

    for index, row in tqdm(enumerate(source.iloc), total=len(source)):
        context = row[3]
        context = convert_new_sbj_obj(context)
        context = remove_space_around_bracket(context)
        context = remove_square_bracket(context)

        sbj, obj, label = convert_new_label_to_origin(row[0].replace('_', ' '), row[1].replace('_', ' '), row[2])
        new_row = {
            "source_code": f"kor-re-{index:05d}",
            "context": context.replace('_', ' '),
            "sbj_entity": sbj,
            "obj_entity": obj,
            "label": label,
        }
        if new_row["label"] != "관계_없음" and no_append_for_none:
            target = target.append(new_row, ignore_index=True)
    
    return target

def convert_new_label_to_origin(sbj: str, obj: str, label: str):
    # 하드 코딩
    # raw 데이터 수정 사항도 있으니 주의
    arg1 = sbj
    arg2 = obj

    if label == "album":
        label = "단체:상위_단체"
    elif label == "artist" or \
            label == "author" or \
            label == "notableWork":
        arg1 = obj
        arg2 = sbj
        label = "인물:제작"
        # 원 데이터에서 가수를 모두 인물로 처리
    elif label == "award" or \
            label == "basedOn" or \
            label == "battle" or \
            label == "commander" or \
            label == "computingPlatform" or \
            label == "field" or \
            label == "genre" or \
            label == "hubAirport" or \
            label == "influenced" or \
            label == "influencedBy" or \
            label == "instrument" or \
            label == "knownFor" or \
            label == "language" or \
            label == "languageFamily" or \
            label == "manufacturer" or \
            label == "officialLanguage" or \
            label == "operatingSystem" or \
            label == "opponent" or \
            label == "pastMember" or \
            label == "place" or \
            label == "predecessor" or \
            label == "previousWork" or \
            label == "recordedIn" or \
            label == "subsequentWork" or \
            label == "successor" or \
            label == "tenant" or \
            label == "unknown" or \
            label == "vicePresident" or \
            label == "recordLabel":
        label = "관계_없음"
    elif label == "birthCountry":
        label = "인물:출생_국가"
    elif label == "birthCity":
        label = "인물:출생_도시"
    elif label == "capital" or \
            label == "region_city" or \
            label == "largestCity":
        label = "단체:본사_도시"
        # 수도를 국가의 소속으로 봐야되니? 이해는 안 되지만...
    elif label == "channel" or \
            label == "developer" or \
            label == "distributor" or \
            label == "industry" or \
            label == "publisher":
        arg1 = obj
        arg2 = sbj
        label = "단체:제작"
    elif label == "child":
        label = "인물:자녀"
    elif label == "city" or \
            label == "garrison" or \
            label == "ground" or \
            label == "locatedInArea_city" or \
            label == "location_city":
        label = "단체:본사_도시"
        # 뭔가 본사 도시는 obj가 도시면 다 해당하는 느낌이다
    elif label == "club" or \
            label == "team" or \
            label == "youthClub":
        label = "인물:소속단체"
        # 정치인의 경우 인물 출신성분에 들어가는 경향이 더 많다.
    elif label == "composer" or \
            label == "director" or \
            label == "musicalArtist" or \
            label == "musicalBand" or \
            label == "producer" or \
            label == "starring" or \
            label == "writer":
        arg1 = obj
        arg2 = sbj
        label = "인물:제작"
    elif label == "country" or \
            label == "garrison_country" or \
            label == "ground_country" or \
            label == "headquarter" or \
            label == "populationPlace":
        label = "단체:본사_국가" 
        # 뭔가 본사 국가는 obj가 국가면 다 해당하는 느낌이다   
    elif label == "currentMember" or \
            label == "ideology" or \
            label == "keyPerson" or \
            label == "leaderName" or \
            label == "owner":
        label = "단체:구성원"
    elif label == "deathCountry":
        label = "인물:사망_국가"
    elif label == "deathPlace":
        label = "인물:사망_도시"
    elif label == "education":
        label = "인물:학교"
    elif label == "foundedBy":
        label = "단체:창립자"
    elif label == "largestCity_state":
        arg1 = obj
        arg2 = sbj
        label = "단체:본사_주(도)"
    elif label == "league":
        label = "단체:상위_단체"
    elif label == "locatedInArea" or \
            label == "location_state" or \
            label == "region":
        label = "단체:본사_주(도)"
        # region은 개념적으로는 다르지만 문장들이 대체로 주, 도를 가리킴
    elif label == "managerClub" or \
            label == "nationalTeam" or \
            label == "owningOrganisation" or \
            label == "party":
        label = "인물:소속단체"
    elif label == "nationality":
        label = "인물:출신성분/국적"
    elif label == "occupation":
        label = "인물:직업/직함"
    elif label == "operator" or \
            label == "owner_group" or \
            label == "owningOrganisation" or \
            label == "sourceMountain":
        label = "단체:상위_단체"
    elif label == "parent":
        label = "인물:부모님"
    elif label == "parentCompany":
        label = "단체:모회사"
    elif label == "product":
        label = "단체:제작"
    elif label == "religion":
        label = "단체:정치/종교성향"
    elif label == "relative":
        label = "인물:기타_친족"
    elif label == "residence":
        label = "인물:거주_국가"
    elif label == "residence_state":
        label = "인물:거주_주(도)"
    elif label == "restingPlace":
        label = "인물:사망_도시"
    elif label == "routeStart" or \
            label == "routeEnd":
        label = "단체:하위_단체"
    elif label == "spouse":
        label = "인물:배우자"
    elif label == "death_reason":
        label = "인물:사망_원인"
    
        
        # 근데 정치인은 소속단체라 해도 될 것 같다.
        # 레이블 구분이 명확하지 않음
        # 그냥 국가면 출신성분/국적으로...

        
    return arg1, arg2, label

def refine_special_letter(source: pd.DataFrame, print_not_refined: bool=False):
    # 특수 기호 통일 및 치환 (위키식 대괄호 삭제)
    counter = 0
    for index in range(len(source)):
        text_origin = source.at[index, "context"]
        text = text_origin
        text = re.sub("[☎☏◕③♤Ⓐ★□♡→•●]", "", text)
        text = re.sub("[∙・･․‧⋅ㆍ·▲▵▴△▶]", ",", text)
        text = re.sub("[。]", "", text)
        text = re.sub("[`‘’´]", "'", text)
        text = re.sub("[∼]", "~", text)
        text = re.sub("[⟪⟫『』《》〈〉「」｢｣“”]", "\"", text)
        text = re.sub("[ℓ]", "리터", text)
        text = re.sub("[㎞]", "km", text)
        text = re.sub("[㎢]", "km²", text)
        text = re.sub("[¼]", "1/4", text)
        text = re.sub("[㎏]", "kg", text)
        text = re.sub("[：]", ":", text)
        text = re.sub("[㎡]", "m²", text)
        text = re.sub("[ｍ]", "m", text)
        text = re.sub("[–─―↔]", "-", text)
        text = re.sub("[／]", "/", text)
        text = re.sub("[％]", "%", text)
        text = re.sub("[＆]", "&", text)
        text = re.sub("[（\xa0]", "", text)
        text = remove_square_bracket(text)

        # text = re.sub("|", " ", text)
        source.at[index, "context"] = text
        if text != text_origin:
            counter += 1

    print(f"Refined {counter} texts")
    if print_not_refined:
        # 허용되는 특수 기호 외 다른 기호가 있는지 확인하는 코드, 필요 없음.
        for index, row in enumerate(source.iloc):
            text = row["context"]
            result = re.search(
                r"""[^\w !@#$%^*&()_=+\\\/|\[\]{};:'",.<>?€£㈜\~\-¹²³⁴]""",
                text
            )
            if result:
                print(text)
                print(result)
                print()
    
    # 대괄호로 묶인 부분 확인하는 코드, 필요 없음.
    # for index, row in enumerate(source.iloc):
    #     text = row["context"]
    #     result = re.search(
    #         r"""\{""",
    #         text
    #     )
    #     if result:
    #         print(text)
    #         print(result)
    #         print()

def convert_new_sbj_obj(target: str) -> str:
    """새 데이터셋의 sbj, obj 토큰을 변환

    ex.) ^[[_sbj_]]^ ==> {{_sbj_}}

    Args:
        target (str): 원래 문장

    Returns:
        str: 변환된 문장
    """
    return re.sub(r"\s\[\[\s(_obj_|_sbj_)\s\]\]\s", r"{{\1}}", target)

def remove_space_around_bracket(target: str) -> str:
    """괄호 주변의 띄어쓰기를 제거

    ex.) ^[[^XXXX^]]^ ==> [[XXXX]]

    Args:
        target (str): 원래 문장

    Returns:
        str: 변환된 문장
    """
    return re.sub(r"\s(\[\[|\]\])\s", r"\1", target)

def remove_square_bracket(target: str) -> str:
    """위키형태의 연속된 대괄호로 된 단어를 원래대로 변환

    ex.) ^[[_sbj_]]^ ==> {{_sbj_}}

    Args:
        target (str): 원래 문장

    Returns:
        str: 변환된 문장
    """
    return re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]", r"\1", target)
    # \[\[      ==> [[ 괄호로 시작하는 텍스트
    # (?:       ==> 추적하지 않는 그룹
    # [^\]|]*   ==> ] 또는 |를 포함하지 않는 텍스트(연속 글자)
    # \|        ==> |로 끝나는
    # )         ==> 그룹의 끝
    # ?         ==> 앞의 그룹이 있거나 없거나
    # (         ==> 추적하는 그룹
    # [^\]|]*   ==> ] 또는 |를 포함하지 않는 텍스트(연속 글자)
    # )         ==> 그룹의 끝
    # \]\]      ==> ]] 괄호로 닫는 텍스트


def check_label_difference(source: pd.DataFrame, print_bad_label: bool=False):
    # 데이터의 sbj, obj 위치와 실제 위치의 차이 체크
    counter_sbj = 0
    counter_obj = 0
    no_sbj = 0
    no_obj = 0
    for index, row in enumerate(source.iloc):
        text: str = row["context"]
        sbj = row["sbj_entity"]
        sbj_start = row["sbj_str"]
        obj = row["obj_entity"]
        obj_start = row["obj_str"]
        
        ssp = sbj_start - 2
        osp = obj_start - 2
        fs_start = text.find(sbj, ssp if ssp >= 0 else 0)
        fo_start = text.find(obj, osp if osp >= 0 else 0)

        if fs_start == -1:
            no_sbj += 1
        if fo_start == -1:
            no_obj += 1

        do_lf = False
        if fs_start != sbj_start:
            counter_sbj += 1
            if print_bad_label:
                print(f"{index} sbj: {abs(fs_start - sbj_start)}")
                print(text)
                print(f"{sbj} --> Label: {sbj_start}, Finded at {fs_start}")
                do_lf = True
        if fo_start != obj_start:
            counter_obj += 1
            if print_bad_label:
                print(f"{index} obj: {abs(fo_start - obj_start)}")
                print(text)
                print(f"{obj} --> Label: {obj_start}, Finded at {fo_start}")
                do_lf = True
        if do_lf:
            print()
    
    print("Position Label Check Result")
    print(f"sbj diff: {counter_sbj}")
    print(f"obj diff: {counter_obj}")
    print(f"fail to find sbj: {no_sbj}")
    print(f"fail to find obj: {no_obj}")


if __name__ == "__main__":
    main()
