import setuptools

setuptools.setup(name='torch_trainer',
                 version='1.0',
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),
                 )
