# FastAPI Documentation

Source: https://fastapi.tiangolo.com/tutorial/

## First Steps
To create a basic FastAPI application, import `FastAPI` and create an instance.
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```
Run the application with Uvicorn:
`uvicorn main:app --reload`
The `--reload` flag enables auto-reload on code changes and should only be used during local development.

## Path Parameters
You can declare path parameters or variables with the same syntax used by Python format strings.
```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```
The value of `item_id` will be passed to your function as the argument `item_id`. Data validation is handled automatically using Python type hints.

## Query Parameters
When you declare other function parameters that are not part of the path parameters, they are automatically interpreted as "query" parameters.
```python
@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```
Query parameters are key-value pairs that appear after the `?` in a URL, separated by `&` characters.

## Request Body and Pydantic Models
When you need to send data from a client to your API, you send it as a request body. To declare a request body, you use Pydantic models with all their power and benefits.
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
def create_item(item: Item):
    return item
```
FastAPI will read the body of the request as JSON, convert types, and validate the data.

## Dependency Injection
FastAPI has a powerful Dependency Injection system. Dependencies are declared using `Depends`.
```python
from fastapi import Depends

def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/users/")
def read_users(commons: dict = Depends(common_parameters)):
    return commons
```
This allows code reuse, shared database connections, and enforced security requirements across multiple endpoints.

## Background Tasks
You can define background tasks to run after returning a response. This is useful for operations like sending email notifications or processing files where the client doesn't need to wait for completion.
```python
from fastapi import BackgroundTasks

def write_notification(email: str, message: str = ""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)

@app.post("/send-notification/{email}")
def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

## Error Handling and HTTPException
To return HTTP error responses with status codes, raise an `HTTPException`.
```python
from fastapi import HTTPException

@app.get("/items-error/{item_id}")
def read_item_error(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```
You can also install custom exception handlers using `@app.exception_handler(CustomException)`.
