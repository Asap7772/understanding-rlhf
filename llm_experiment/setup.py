from setuptools import setup, find_packages

setup(
    name='understanding_rlhf',
    version='0.1',
    packages=find_packages(),
    description='A Python package for a conservative reward model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Anikait Singh',  # Replace with your name
    author_email='anikait@stanford.edu',  # Replace with your email
    install_requires=[
        'torch',  # This is the package name for PyTorch
    ],
    classifiers=[
        # It's good practice to specify compatible Python versions and other metadata here
        # Full list at https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your choice of license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify your required Python version
)
