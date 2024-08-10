import asyncio
import aiohttp
import pandas as pd
import json
import io
from aiofiles import open as aio_open

API_KEY = None
FILE_PATH = "data/남구/남구_crack_Mobiltech_data.csv"

def get_analysis_function_schema():
    return {
        "name": "analyze_data",
        "description": "Analyze road damage data and provide insights and recommendations.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "description": "The data to be analyzed in JSON format.",
                    "items": {
                        "type": "object",
                        "properties": {
                            # 여기에 데이터 필드를 정의할 수 있습니다
                            "field1": {"type": "string"},
                            "field2": {"type": "number"},
                            # 데이터 필드에 맞게 더 추가할 수 있습니다
                        }
                    }
                }
            },
            "required": ["data"]
        }
    }

async def build_data_chunk(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dictionaries (JSON serializable)."""
    data = df.to_dict(orient='records')
    return data

async def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Read data from file and return DataFrame."""
    async with aio_open(file_path, mode='r') as f:
        content = await f.read()
        df = pd.read_csv(io.StringIO(content))
        return df

async def fetch_analysis(session: aiohttp.ClientSession, data_chunk: list) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",  # 최신 모델 사용
        "messages": [
            {"role": "system", "content": "You are an expert in data analysis. Provide detailed analysis and recommendations based on the given data."},
            {"role": "user", "content": "Analyze the following data and provide insights and recommendations according to the given prompts."}
        ],
        "functions": [get_analysis_function_schema()],
        "function_call": "auto",  # 자동으로 함수 호출
    }
    
    # 데이터를 분석 함수에 올바르게 전달
    function_arguments = {"data": data_chunk}
    payload["function_call"] = {"name": "analyze_data", "arguments": json.dumps(function_arguments)}

    async with session.post(url, headers=headers, json=payload) as response:
        response_json = await response.json()
        return response_json


def build_summary_prompt(analysis_results: str) -> str:
    return f"""
다음 데이터는 포항 도로 손상 사건에 대한 정보입니다:
{analysis_results}
------ 분석 시작 -----
다음 항목에 대해 분석과 제안을 제공해 주세요:
1. 패턴 및 트렌드 분석: 손상 사건이 특정 지역이나 시간대에 집중되는지 분석할 것
2. 위치 기반 정보 제공: 각 사건의 위치를 분석하여 지역별 위치 정보를 제공할 것
3. 몇시에 언제 몇월에 가장 많이 발생하는지 정보 요약
4. 어디에서 도로가 손상될지 예측해볼 것

각 항목에 대해 구체적이고 명확한 분석 및 제안을 제공할 것
------ 분석 끝 -------
"""

async def fetch_summary(session: aiohttp.ClientSession, summary_prompt: str) -> str:
    """Fetch summary from OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are an expert in summarizing data analysis results. Provide a concise and comprehensive summary based on the provided analysis results."},
            {"role": "user", "content": summary_prompt}
        ],
        "max_tokens": 1500
    }
    async with session.post(url, headers=headers, json=payload) as response:
        response_json = await response.json()
        return response_json['choices'][0]['message']['content']

async def main():
    df = await load_and_prepare_data(FILE_PATH)
    data_chunk = await build_data_chunk(df)

    async with aiohttp.ClientSession() as session:
        # Fetch the analysis
        analysis_response = await fetch_analysis(session, data_chunk)
        print(analysis_response)
        # analysis_result = analysis_response['choices'][0]['message']['content']
        # # # Build summary prompt
        # summary_prompt = build_summary_prompt(analysis_result)

        # # Fetch the summary
        # summary = await fetch_summary(session=session, summary_prompt=summary_prompt)
        
        # print("최종 요약:")
        # print(summary)

if __name__ == "__main__":
    asyncio.run(main())
