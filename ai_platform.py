import asyncio
import aiohttp
import pandas as pd
import json
import io
from aiofiles import open as aio_open

# OpenAI API 키 설정 (환경 변수로 설정하는 것이 좋습니다)
API_KEY = None
FILE_PATH = "data/Mobiltech_crack_data.csv"

async def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    async with aio_open(file_path, mode='r') as f:
        content = await f.read()
        df = pd.read_csv(io.StringIO(content))
        return df

def build_prompt(data: list) -> str:
    # 데이터 리스트를 JSON 문자열로 변환
    data_chunk = json.dumps(data, ensure_ascii=False, indent=2)
    return f"""
다음 데이터는 포항 도로 손상 사건에 대한 정보입니다:
{data_chunk}
------ 분석 시작 -----
다음 항목에 대해 분석과 제안을 제공해 주세요:
1. 패턴 및 트렌드 분석: 손상 사건이 특정 지역이나 시간대에 집중되는지 분석할것
2. 위치 기반 정보 제공: 각 사건의 위치를 분석하여 지역별 위치 정보를 제공할것
3. 몇시에 언제 몇월에 가장 많이 발생하는지 정보 요약
4. 어디에서 도로가 손상될지 예측해볼것

각 항목에 대해 구체적이고 명확한 분석 및 제안을 제공할것
------분석 끝 -------
"""

async def fetch_analysis(session: aiohttp.ClientSession, prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are an expert in data analysis and problem-solving. Provide detailed analysis and recommendations based on the given data."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000  # 필요에 따라 조정
    }
    async with session.post(url, headers=headers, json=payload) as response:
        response_json = await response.json()
        return response_json['choices'][0]['message']['content']

async def analyze_data_in_batches(df: pd.DataFrame, batch_size: int = 100) -> list:
    # 데이터를 배치로 나누어 처리
    async with aiohttp.ClientSession() as session:
        tasks = []
        for start in range(0, len(df), batch_size):
            batch = df[start:start + batch_size]
            data_chunk = batch.to_dict(orient='records')
            prompt = build_prompt(data_chunk)
            task = fetch_analysis(session, prompt)
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

def build_summary(analyses: list) -> str:
    # 각 분석 결과를 요약
    combined_text = "\n".join(analyses)
    summary_prompt = f"""
다음은 여러 분석 결과를 종합한 내용입니다:
{combined_text}

위 내용을 바탕으로 바로 본론으로 최종 요약을 제공해 주세요: 
1. 전체 분석 결과 요약
2. 주요 패턴 및 트렌드 요약
3. 지역별 도로 손상 정보 요약
4. 몇시에 언제 몇월에 가장 많이 발생하는지 정보 요약
5. 권장 사항 및 개선 방안 요약

최종 요약을 간결하고 명확하게 작성해 주세요.
"""
    return summary_prompt

async def fetch_summary(session: aiohttp.ClientSession, prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an expert in summarizing and synthesizing information. Provide a concise summary based on the given data."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500  # 필요에 따라 조정
    }
    async with session.post(url, headers=headers, json=payload) as response:
        response_json = await response.json()
        return response_json['choices'][0]['message']['content']

async def main():
    # 데이터 로드 및 분석 수행
    df = await load_and_prepare_data(FILE_PATH)
    analysis_results = await analyze_data_in_batches(df)
    
    # 분석 결과를 종합하여 최종 요약 생성
    summary_prompt = build_summary(analysis_results)
    async with aiohttp.ClientSession() as session:
        final_summary = await fetch_summary(session, summary_prompt)
        print("최종 요약:")
        return final_summary
    
if __name__ == "__main__":
    asyncio.run(main())
