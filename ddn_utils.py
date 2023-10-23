import numpy

try:  # Compatible with s3
    import brainpp_yl.fs

    brainpp_yl.fs.compat_mode()
except:
    pass
