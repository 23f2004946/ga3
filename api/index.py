from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

class CommentRequest(BaseModel):
    comment: str


@app.post("/comment")
async def analyze_comment(data: CommentRequest):

    if not data.comment.strip():
        raise HTTPException(status_code=400, detail="Empty comment")

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",

            messages=[
                {"role":"system",
                 "content":"Analyze sentiment and give rating 1-5"},
                {"role":"user","content":data.comment}
            ],

            response_format={
                "type":"json_schema",
                "json_schema":{
                    "name":"sentiment_analysis",
                    "schema":{
                        "type":"object",
                        "properties":{
                            "sentiment":{
                                "type":"string",
                                "enum":["positive","negative","neutral"]
                            },
                            "rating":{
                                "type":"integer",
                                "minimum":1,
                                "maximum":5
                            }
                        },
                        "required":["sentiment","rating"]
                    }
                }
            }
        )

        return json.loads(
            response.choices[0].message.content
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
