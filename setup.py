from setuptools import setup, find_packages

setup(
    name='NenuPlot',
    version='4.0',
    python_requires='>=2.7',
    description='NenuPlot is a tool designed to assist in the visualization and analysis of PSRFITS folded files. It provides a quick-look generation in PDF and PNG formats and offers numerous options for data handling, including cleaning, rebinding, RM and DM fitting, and data extraction within specified frequency and time ranges.',
    author='Louis Bondonneau',
    author_email='louis.bondonneau@obs-nancay.fr',
    # packages=find_packages(),
    packages=find_packages(),
    package_data={"NenuPlot_module": ["NenuPlot.conf"]},
    install_requires=[
        'numpy',
        'astropy',
        'matplotlib'
    ],
    scripts=["scripts/NenuPlot.py",
             ],
)
