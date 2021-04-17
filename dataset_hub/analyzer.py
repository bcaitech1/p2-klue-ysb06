import pandas as pd
import re

def main():
    raw_origin: pd.DataFrame = pd.read_csv("/opt/ml/input/data/train/train.tsv", delimiter='\t', header=None)
    raw_origin.columns = ["source_code", "context", "sbj_entity", "sbj_str", "sbj_end", "obj_entity", "obj_str", "obj_end", "label"]
    raw_target: pd.DataFrame = pd.read_csv("/workspace/kor-re-gold/gold-standard-v1/agreement_content.txt", delimiter='\t', header=None)

    # sorted_raw = raw_origin.sort_values(by="context")
    # sorted_raw = sorted_raw.reset_index()
    # refine_special_letter(sorted_raw.loc)
    refine_special_letter(raw_origin.loc)


def refine_special_letter(source):
    print("\n")
    for index, row in enumerate(source):
        text = row["context"]
        result = re.search("[^A-Za-z0-9가-힣一-龥ぁ-ゔァ-ヴー々〆〤 ,.\[\]():;“”%&'\"\-~]", text)
        if result:
            print(text)
            print(result)
            print()
        if index == 1000:
            break
    print("\n")





if __name__ == "__main__":
    main()