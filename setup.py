from setuptools import setup, find_packages
from pathlib import Path

readme = Path(__file__).parent / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="voxpolish",
    version="0.1.0",
    description="VoxPolish — a TTS dataset cleaner. Remove noise, trim silence, and normalize loudness for text-to-speech training data in one command.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jeeva",
    url="https://github.com/Jeevav62/voxpolish",
    keywords=[
        "tts", "tts-dataset", "dataset-cleaner", "text-to-speech",
        "speech-enhancement", "audio-denoising", "noise-reduction",
        "audio-processing", "dataset-cleaning", "voice", "deep-learning",
    ],
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "deepfilternet>=0.5.6",
        "torch>=2.0,<3.0",
        "torchaudio>=2.0,<3.0",
        "soundfile>=0.10",
        "pyloudnorm>=0.1.0",
        "gradio>=6.0",
        "tqdm>=4.0",
    ],
    entry_points={
        "console_scripts": [
            "voxpolish=clean:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
)
