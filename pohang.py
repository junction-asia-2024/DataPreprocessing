import os
import pandas as pd
import numpy as np

# 데이터 로드
data = pd.read_csv("data/경상북도.csv")

# 확인할 컬럼
drop_kind_data = ["geom", "utm_x", "utm_y", "utm_z", "status"]

# 재배열할 컬럼
reindex_col = ["id", "longitude", "latitude", "classname", "time", "address"]

# null 값을 가진 행을 삭제한 새로운 DataFrame 생성
filtered_data = data.drop(columns=drop_kind_data)

# 'gu' 열 생성: regionname_2의 값을 안전하게 처리하여 두 번째 요소만 추출
filtered_data["gu"] = filtered_data["regionname_2"].apply(
    lambda x: x.split(" ")[1] if isinstance(x, str) and len(x.split(" ")) > 1 else None
)
filtered_data["time"] = filtered_data["time"].str.split(".").apply(lambda x: x[0])

# 확인할 열 이름들
region_names = [
    "regionname_1",
    "regionname_2",
    "regionname_3",
    "regionname_4",
    "roadaddr",
    "roadaddrnum",
]


def data_combined(dataframe: pd.DataFrame) -> None:
    concat_data = []
    
    # 'gu'와 'classname'의 고유값 추출
    unique_gu = np.unique(dataframe["gu"].dropna().values)
    unique_value = np.unique(dataframe["classname"].dropna().values)

    for i in unique_gu:
        for col in unique_value:
            # 'classname'이 col 값이면서 'gu'가 i 값인 데이터만 필터링
            filtered_data = dataframe[
                (dataframe["classname"] == col) & (dataframe["gu"] == i)
            ]
            # `regionname_4` 열의 존재 여부 확인
            if "regionname_4" in filtered_data.columns:
                # `regionname_4`가 존재하면 모든 열을 결합
                filtered_data["address"] = (
                    data[region_names]
                    .astype(str)
                    .apply(
                        lambda row: " ".join(
                            [
                                str(item)
                                for item in row
                                if item != "nan" and item.strip() != ""
                            ]
                        ),
                        axis=1,
                    )
                )
            else:
                # `regionname_4`가 존재하지 않으면 `regionname_1`, `regionname_2`, `regionname_3`만 결합
                filtered_data["address"] = (
                    filtered_data[["regionname_1", "regionname_2", "regionname_3"]]
                    .astype(str)
                    .apply(
                        lambda row: " ".join(
                            [
                                str(item)
                                for item in row
                                if item != "nan" and item.strip() != ""
                            ]
                        ),
                        axis=1,
                    )
                )

            filtered_data.drop(columns=region_names, inplace=True)

            if not filtered_data.empty:
                print(f"Data for classname = {col} and gu = {i}:")

                # 'gu' 열 삭제
                filtered_data = filtered_data.drop(columns=["gu"])

                filtered_data.reindex(columns=reindex_col)
                concat_data.append(filtered_data)
    
    return concat_data

# 데이터 로드 및 예제 실행
data = data_combined(filtered_data)
pd.DataFrame(pd.concat(data)).to_csv("total_pohang.csv", index=False)
