from setuptools import setup, find_packages

setup(
    name="AS_LLM_nodes",
    version="1.0.0",
    description="A ComfyUI extension providing Gemini and ChatGPT integration via custom nodes.",
    author="Artem Svetozarov",
    author_email="art.svetozarov@gmail.com",
    url="https://github.com/svetozarov/AS_LLM_nodes",
    packages=find_packages(),
    install_requires=[
        "Pillow",
        "requests",
        "google-generativeai",
        "openai"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
