# Converter of the quilt image into the native Looking Glass format
This program can be used to convert the input quilt image into the native Looking Glass format. The result can be displayed directly on the Looking Glass in a fullscreen mode. The display calibration parameters are necessary to be defined. They can be obtained by using [this tool](https://github.com/ichlubna/getLKGCalibration) or in the json file in the Looking Glass Portrait mounted memory. Use `--help` for the description of the parameters. The program uses OpenCL for GPU acceleration. Make sure to have the `kernel.cl` code in the same directory as the binary.

Example:
```
./QuiltToNative -i quilt.jpg -o outputDir -cols 8 -rows 6 -width 1536 -height 2048 -pitch 246.867 -tilt -0.185828 -center 0.350117 -viewPortion 1 -subp 0.000217014 -focus 0
```
