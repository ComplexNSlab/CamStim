from setuptools import setup, Extension

setup(
    name="cgrabcallback",
    ext_modules=[
        Extension(
            "cgrabcallback",
            ["cgrabcallback.c"],
        )
    ],
)
