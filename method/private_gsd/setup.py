from setuptools import setup

setup(

    name='Private-GSD',
    version='1.00',
    description='Implementation of Private-GSD mechanism.',
    url='https://github.com/giusevtr/private_gsd.git',
    author='Giuseppe Vietri, Jingwu Tang, Terrence Liu',
    license='MIT',
    packages=['dev', 'models', 'stats', 'utils'],
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn',
                      'tqdm', 'matplotlib', 'seaborn', 'chex', 'diffprivlib'],
)