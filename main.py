from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------- Request Schema --------

class CommentRequest(BaseModel):
    comment: str


# -------- Response Schema --------

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(data: CommentRequest):

    # Input validation
    if not data.comment.strip():
        raise HTTPException(
            status_code=400,
            detail="Comment cannot be empty"
        )

    try:

        response = client.chat.completions.create(
            model="gpt-4.1-mini",

            messages=[
                {
                    "role": "system",
                    "content":
                    """
                    Analyze customer feedback sentiment.

                    sentiment must be:
                    positive, negative, neutral

                    rating:
                    5 = highly positive
                    4 = positive
                    3 = neutral
                    2 = negative
                    1 = highly negative
                    """
                },
                {
                    "role": "user",
                    "content": data.comment
                }
            ],

            # ✅ Structured Outputs JSON Schema
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": [
                                    "positive",
                                    "negative",
                                    "neutral"
                                ]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": [
                            "sentiment",
                            "rating"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        )

        result = response.choices[0].message.content

        import json
        parsed = json.loads(result)

        return parsed

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {str(e)}"
        )