from pathlib import Path
import setuptools

install_requires = []

lib_dir = Path(__file__).parent
requirements_file = lib_dir / 'requirements.txt'
with open(requirements_file) as stream:
    install_requires += stream.read().splitlines()

setuptools.setup(
    name='torchhep',
    project_name='TorchHEP',
    version='0.0.1',
    author='Seungjin Yang',
    author_email='slowmoyang@gmail.com',
    url='https://github.com/slowmoyang/TorchHEP',
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=install_requires
)
