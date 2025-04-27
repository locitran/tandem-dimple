from distutils.core import setup, Extension

setup(
  name='PyRONN',
  packages=['ronn'],
  version='4.0rc0',
  description='A refactoring/reimplementation of RONN with a Python extension!',
  author='Shyam Saladi',
  author_email='saladi@caltech.edu',
  url='https://github.com/smsaladi/pyronn',
  download_url='https://github.com/smsaladi/pyronn/tarball/4.0',
  keywords=['protein', 'disorder', 'sequence', 'bioinformatics'],
  license='Non-commercial Academic Use License',
  ext_modules=[Extension('ronn.libronn',
                         sources=['ronn/libRONN.cpp',
                                  'ronn/callBBF.cpp'])],
  package_data={'ronn': ['data/c*/model.rec', 'data/c*/pdfs.rec']},
  entry_points={'console_scripts': ['RONN=ronn.ronn:main']},
  zip_safe=True,
  include_package_data=True,
  setup_requires=['pytest-runner'],
  tests_require=['pytest', 'pandas']
)
