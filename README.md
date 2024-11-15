# ACCESS-Vis Python Visualisation Package for ACCESS Models.

[![DOI](https://zenodo.org/badge/767301983.svg)](https://doi.org/10.5281/zenodo.14167608)

[![Documentation Status](https://readthedocs.org/projects/access-vis/badge/?version=latest)](https://access-vis.readthedocs.io/en/latest/?badge=latest)

Visualisation examples and resources, including open-source 3D vis for ACCESS-NRI releases

This repository is intended as a resource for complex visualisation tasks using ACCESS models and other related data sources.

What this space for published source code for our released visualisations which will be added here as soon as tested and ready for public release.

Included the python module 'accessvis' which can be installed from here with `python -m pip install --editable .`

See also the included `install.sh` to set up the jupyter kernels for running in ARE on gadi.

Here's an example of how you can include versioning instructions in your README file, explaining the use of tags in the `v0.0.0` format:

---

### Versioning

We use semantic versioning for this project, following the format `vX.Y.Z`, where:

- **X** (Major version): This number is incremented when there are significant changes, breaking backward compatibility. For example, upgrading the project to a new technology or introducing major new features that alter the way the project works.
  
- **Y** (Minor version): This number is incremented when new functionality is added in a backward-compatible manner. For instance, adding new features, modules, or improvements that don’t disrupt existing functionality.
  
- **Z** (Patch version): This number is incremented for bug fixes, security patches, or small improvements that do not affect the overall functionality of the project.

When creating a new release, make sure to tag it with the appropriate version number in the `vX.Y.Z` format. For example, to tag version `v1.2.0`, use the following Git command:

```bash
git tag v1.2.0
git push origin v1.2.0
```

### Config and DATA caching

By default, `ACCESS-Vis` caches its data in different locations depending on the platform. On NCI Gadi, data are stored in `/scratch/$PROJECT/$USER/.accessvis`, while on other platforms, the data are cached in `$HOME/.accessvis`. However, users can customize this path by setting the `ACCESSVIS_DATA_DIR` environment variable to a directory of their choice.


