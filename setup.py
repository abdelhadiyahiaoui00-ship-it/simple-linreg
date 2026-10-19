from setuptools import setup

setup(
    name='simple-linreg-abdelhadi',,
    version='0.1.0',
    author='Your Name',
    author_email='you@example.com',
    packages=['simple_linreg'],
    url='http://pypi.python.org/pypi/simple-linreg/',
    license='MIT',
    license_files=['LICENSE.txt'],
    description='Simple linear regression implemented from scratch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    install_requires=['numpy'],
)
