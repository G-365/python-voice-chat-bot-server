from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

import wave
import socketio
import requests
import nltk

nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Allow frontend running locally
    "http://192.168.133.171:3000",
    "https://95aa-149-7-16-247.ngrok-free.app",  # Replace with your actual ngrok URL
]

# Google Cloud Speech Client setup
speech_client = speech.SpeechClient(
    credentials={
        "type": "service_account",
        "project_id": "radar070324",
        "private_key_id": "8ad7c89aeaa8166c2cba3a1a78ffa137dda75f94",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCmggwMen/YacRQ\n1WGGHFecPg3gjTEmQ8cGdRFSIIOBmkAZp9g9cT4UWkk4emC0xQlLMTYjQrWxejMk\nPby4NjE6ttcTvKEocUP3L/bA99tzdFx0og4gxZaU9Z766uZ6hHTZw62DGHqJOxYw\nPyStlcGmnz+D46JPz6zcRP2axwy22DFrZz6fHjzXPtNxlDfrdCkMhRFtMEs3euHf\ns7MGNXSpN8EP2PPAuaQ/G4gWWlHPqUxWGLD0mOvEk3KtnRdYQs/Mdh1XsdPjcpZa\n5AcPQ3tCQXAQOYBcCCChSGHWWVcdQQ6qIwk/2sL6GLh8cf4UlUDhpq/KMhqYCsfQ\n6lyaKSuzAgMBAAECggEATvXTzeUXlGKPwL93yHfPSh4ZSZfbK2ivzI2egaI+iqrB\n5Ai24Gg/xroMB/bsvjzEC/7RzAXaEMhA0VpfkMHONag4NTlZ+UpBL0r1CoxfapBP\nOdYRuhPJNWmHEzlqw8XlfdEwCr+EeGhnPMjs1U5zr1bMcXh048E5mZkz5H6pNllq\nqeLRVLH5M8BNU3lNL4kh0/TvB8oAtS19M6JpR02IcO7divamtuLzkZYVyll0YG8g\nriW7T2HlEz+N4dfnZLSQobie1RR8IsjdruBVkQkWc14c7w0I/kvAMhZwT5MN5nQJ\nWJ5uyuxD4aEqh4whaXTtIaPYxZfhJ6HfRjokRynmFQKBgQDU3kXidOeF5zCf/+y5\nYvMU5MCJfLJakRyYbbt+9Mxq7yAeoLEeokAENbhMeGVZ5a5U+KxhpWT8K9u39HU6\ntx/LUYZZGU3hroO8lEEcH7h51dl87X/PnkhXa5+6YZZbluNKcpRAPfGgKrlKvmMi\nIc1paLW077SWQrBgOo4e0UWArwKBgQDIPwNPFHoz73qU03hUsZSeVn24UI/tDpV0\n4hGZcyRmQ5O2NHSmtdczPlOumtZMp7hgMYcaBq08xADN/8mSnPX7XyNX6RM3Mfz6\nToie8+MIVsZN8PXlUQpufY9ts2ksJzTbqKkYdes88pQEj01mhrnutHWduWAf4W5D\n61+C8AsePQKBgEf69WSzJUr5N08TSmgR1qLdC2IyYVkQsru0d29htfH+9DyHF/2E\n3eLOi4iIObVhXkbrY9cNB43iAsU8i5uUKtMkuSpNzTEgQvm0pCOvckD4mDePU+XP\n3yR4hyWONDq6VhdpkUn76EXBzLBCmuECzyPyvWb2m3koCd4wTriLCVaPAoGATKSx\no4b27wHuLSBzohcGB0Sbgfxz2gwG8GHG0rDbcbjTxJ13OIfJAngMl3v0Igrf6xGJ\n4FBF5kgu9qm8gT1KeRgE8xTmoe2kIjoE2LIIZ5yu8g4UT90g3QF58EcHLjsjZB+9\n+PrJOt6uAMDIo0FV0SOJEQFu5UTna1+fgwJVfxECgYBiQH3d+GYS6X+apMFTqZSa\ntzFiZ4+E9TLwn4GKXpVF/JKqHt2HrxvT6r0okBFNKEFooBlEw9F0GXuChkGe+6R2\nxatlSmreXLTpGXnwM2GuGHty1kIIQcP3luo8nl1IRMxLbsiT7hCyYfthCXdUgXUI\n6c43B+UwaDLkXTCwjuxBIg==\n-----END PRIVATE KEY-----\n",
        "client_email": "radar081224@radar070324.iam.gserviceaccount.com",
        "client_id": "107573680322304636819",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/radar081224%40radar070324.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Allow all origins (change to your frontend URL for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a Socket.IO server with ASGI support
sio = socketio.AsyncServer(
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode="asgi",
    logger=True,
    engineio_logger=True,
)  # Allow all origins (for cross-platform)

# Mount Socket.IO server onto FastAPI
app.mount("/", socketio.ASGIApp(sio, other_asgi_app=app))


@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")


@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")


@sio.on("t_qta")
async def handle_t_qta(sid, data):
    try:
        print(f"Message from {sid}")
        response = await qta(data)
        messages = response[0][0][1]

        await sio.emit("t_response", {"messages": messages}, to=sid)
        sentences = sent_tokenize(messages)
        index = 0
        while index < len(sentences):
            sentence = sentences[index]
            index += 1
            s_response = await gen_speech(sentence)
            await sio.emit("s_response", {"speech": s_response}, to=sid)

    except Exception as e:
        print(f"Handle failed in t_qta: {e}")


@sio.on("s_qta")
async def handle_s_qta(sid, data):
    print(f"Speech from {sid}")
    # Setup the streaming configuration
    streaming_config = get_streaming_config()

    # Streaming Recognizer to process the audio data in real-time
    requests = generate_audio_stream(data)
    streaming_recognizer = speech_client.streaming_recognize(streaming_config, requests)

    for response in streaming_recognizer:
        for result in response.results:
            if result.is_final:
                transcription = result.alternatives[0].transcript
                print(f"Transcription: {transcription}")
                await sio.emit(
                    "transcription", {"transcription": transcription}, to=sid
                )
                sentences = sent_tokenize(transcription)
                index = 0
                while index < len(sentences):
                    sentence = sentences[index]
                    index += 1
                    s_response = await gen_speech(sentence)
                    await sio.emit("s_response", {"speech": s_response}, to=sid)


@app.get("/")
async def root():
    return {"message": "FastAPI + Socket.IO Server Running"}


async def qta(question: str):
    try:
        print(f"Start question to answer")
        client = Client(
            src="on1onmangoes/radarheyzzk250116vGV",
            hf_token="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",
        )

        response = client.predict(
            user_message=question, api_name="/api_get_response_on_submit_button"
        )

        return response
    except Exception as e:
        print(f"Request failed in qta: {e}")


async def gen_speech(sentence: str):
    try:
        response = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/cjVigY5qzO86Huf0OWal",
            json={"text": sentence},
            headers={
                "xi-api-key": "sk_bc1cb48740139b31b087090ba0645d27087c09ee89b3e260",
                "Content-Type": "application/json",
            },
        )
        if response.status_code == 200:
            print(f"Start speach from sentence")
            return response.content
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed in gen_speech: {e}")


# Function to create the streaming configuration
async def get_streaming_config():
    return types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # Set sample rate (matching the frontend settings)
        language_code="en-US",  # Set your language
    )


# This will be the generator that sends audio to Google Cloud in real-time
async def generate_audio_stream(audio_data):
    # Process the received audio data in chunks and yield to Google Cloud's Speech-to-Text API
    audio = types.RecognitionAudio(content=audio_data)
    request = types.StreamingRecognizeRequest(audio=audio)

    yield request


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
