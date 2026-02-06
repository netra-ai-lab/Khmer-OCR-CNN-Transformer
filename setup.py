from setuptools import setup, find_packages

setup(
    name="khmer-ocr",
    version="0.1",
    packages=find_packages(),
    py_modules=["ocr_engine"],
    install_requires=[
        "pillow==9.5.0",
        "numpy",
        "opencv-python",
        "tqdm",
        "datasets",
        "transformers",
        "torch",
        "torchvision",
        "reportlab",
        "python-docx"
    ],
    entry_points={
        'console_scripts': [
            'ocr=ocr_engine:main',  # Format: 'command=module:function'
        ],
    },
)