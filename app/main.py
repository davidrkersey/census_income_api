from fastapi import FastAPI
from routers import predict

app = FastAPI()

# Include the predict router
app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Sound Realty model predictions"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)