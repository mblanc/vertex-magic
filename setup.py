import setuptools, find_packages

setuptools.setup(
    name="vertex-magic",
    version='0.0.1',
    url='https://github.com/mblanc/vertex-magic',
    author="Matthieu Blanc",
    author_email='blanc.matthieu@gmail.com',
    description="A Jupyter Notebook %%magic for training ML Models with Vertex AI",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    license_files = ('LICENSE',),
    install_requires=[
        'ipython',
        'google-cloud-aiplatform',
    ],
    keywords=['ipython', 'jupyter'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ]
)