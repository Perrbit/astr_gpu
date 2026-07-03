# Synchronize after every kernel in first-stage validation

During first-stage GPU validation, every kernel launch is followed by explicit synchronization and error checking. This deliberately sacrifices early performance so NaN generation, illegal memory access, and wrong-kernel failures can be localized before performance optimization removes synchronization points.
