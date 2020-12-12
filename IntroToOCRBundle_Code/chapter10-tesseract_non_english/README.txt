NOTE: Virtual Machine does not include `tessdata` and you would need to follow these instructions to run the code of this chapter.

Steps for adding language packs to Tesseract:

- Step #1: Open terminal and execute the following command to clone the `tessdata` repository: `git clone https://github.com/tesseract-ocr/tessdata`

- Step #2: Change the directory into `tessdata` directory by executing: `cd tessdata`

- Step #3: Determine the full system path of the `tessdata` directory by executing: `pwd`
	- Example: `/home/pyimagesearch/tessdata`

- Step #4: Set the `TESSDATA_PREFIX` environment variable to point to your `tessdata` directory (output of step 3) by executing: `export TESSDATA_PREFIX=/PATH/TO/TESSDATA/REPOSITORY`
	- Example: `export TESSDATA_PREFIX=/home/pyimagesearch/tessdata`