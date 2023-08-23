import asyncio
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI
import time

def cpu_bound_func(param):
    time.sleep(1)
    return param ** param * param


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.executor = ProcessPoolExecutor()
    yield
    app.state.executor.shutdown()


app = FastAPI(lifespan=lifespan)


async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(app.state.executor, fn, *args)  # wait and return result


@app.post("/process")
async def handler(numbers: list):
    numbers = numbers[0].split(",")
    print(numbers, "**************************************************")

    results = await asyncio.gather(*(run_in_process(cpu_bound_func, int(num)) for num in numbers))
    return {"results": results}


@app.post("/process_no_parallel")
async def handler(numbers: list):
    numbers = numbers[0].split(",")
    print(numbers, "**************************************************")
    results = [cpu_bound_func(int(num)) for num in numbers]
    return {"results": results}