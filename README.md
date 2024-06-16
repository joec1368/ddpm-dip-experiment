# DDPM with dpi
## introduction
Using DIP to generate better noise pictures to reduce the time in generating images by DDPM.
## About develop:
I finished this work on colab, then I rewrote the code for local execution.
## prerequire:
1. use 'pip3 install -r requirements.txt' to download all required packages

## how to run:
* there are two files that you can run
1. the first one is DDPM github.ipynb
    * this is the file that I use to do some experiments and development
2. the second one is main.py
    * this is the file that I organized. And it will train ddpm and dip, then it will use dip and ddpm to generate a new picture
    * you can just type 'python3 main.py' to run it!
    * after run the command, you will get the .gif file in directory. You need to open it by yourself.

## report
1. just see report.pdf