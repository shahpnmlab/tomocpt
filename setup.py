import os

import setuptools
from setuptools import setup


def readme():
  readmePath = os.path.abspath(os.path.join(__file__, "..", "README.md"))
  try:
    with open(readmePath) as f:
      return f.read()
  except UnicodeDecodeError:
    try:
      with open(readmePath, 'r', encoding='utf-8') as f:
        return f.read()
    except Exception as e:
      return "Description not available due to unexpected error: "+str(e)

def getVersion():
  initFname = os.path.abspath(os.path.join(__file__, "..", "tomocpt", "__init__.py"))
  import re
  with open(initFname) as f:
      line =  f.readline()
  version = re.match(r"__version__\s+=\s+\"(\d+\.\d+\.\d+)", line).group(1)
  return version

dependency_links = []
install_requires = []
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "requirements.txt"))) as f:
    for line in f:
        if line.startswith("#"):
            continue
        elif line.startswith("--extra-index-url"):
            dependency_links.append(line.strip())
        elif line.startswith("git"):
            line = line.strip()
            package_name = line.split("/")[-1].split("@")[0].split("@")[0]
            install_requires.append( f'{package_name} @ {line}')
        else:
            install_requires.append(line.strip())

setup(name='tomocpt',
      version=getVersion(),
      description='Python tools for picking cryo Tomograms',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='particle picking cryoEM tomograms',
      url='hthttps://github.com/shahpnmlab/tomocpt',
      author='Pranav NM Shah and Ruben Sanchez-Garcia',
      author_email='p.shah.lab@gmail.com',
      license='AGPL-3.0',
      packages=setuptools.find_packages(),
      install_requires=install_requires,
      dependency_links=dependency_links,
      include_package_data=True,
      entry_points={
          'console_scripts': ['tomocpt=tomocpt.main:main',
                              ],
      },
      zip_safe=False
)

