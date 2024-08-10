import os
import openai
from dotenv import load_dotenv
from pd_processing import ranking_danger_combined
load_dotenv()


openai.api_key =  os.getenv("API_KEY")


def generate_summary():
    data = ranking_danger_combined()  

    prompt = (
        f"다음 데이터를 공무원에서 주무관한테 보고하는것 처럼 만들어 주면서 예측도 진행해봐:\n"
        f"{data}\n"  
        f"요약:"
    )

    response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": "You are an assistant that summarizes data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        
    return response.choices[0].message['content'].strip()


def main():    
    # 데이터 요약 생성
    summary = generate_summary()
    return summary


main()