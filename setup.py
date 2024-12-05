from setuptools import setup, find_packages

setup(
    name='tandem',
    version='1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[

    ],
    author='Loci Tran',
    author_email='quangloctrandinh1998vn@gmail.com',
    description='Predicting the pathogenicity of SAVs\nTransfer-leArNing-ready and Dynamics-Empowered Model for DIsease-specific Missense Pathogenicity Level Estimation',
    license='MIT',
    keywords='SAVs, pathogenicity, transfer learning, dynamics',
)