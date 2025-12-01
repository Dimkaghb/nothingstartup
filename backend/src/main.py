import fastapi

app = fastapi.FastAPI()

@app.get("/")
async def read_root():
    return {"server": "Nothing Startup Backend is running!"}
