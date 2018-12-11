from pdf2image import convert_from_path
from os import path
from glob import glob  
import os


# get all files in a given folder path, by a given type of file
# dir_path: path directory 
# ext: the file type, eg: 'pdf'. Find all pdf files in this given folder
def getAllfiles(dir_path, ext):
    tempList = glob(path.join(dir_path,"*.{}".format(ext)))
    tempList += (glob(path.join(dir_path,"*.{}".format(ext.upper()))))
    return tempList


# input a folder path, convert all pdf files in this folder to jpg images
# folder_dir: input a folder path which contains all pdf files
# output_dir: the ouput folder for the output jpg images
def convertPdf2JPG(folder_dir, output_dir):
    # get all pdf files
    pdfFiles = getAllfiles(folder_dir, 'pdf')
    for pdfFile_index, pdfFile in enumerate(pdfFiles):
        image = convert_from_path(pdfFile, fmt='jpg')
        image = image[0]
        index1 = pdfFile.rfind('/')
        index2 = pdfFile.rfind('.')
        pdfFileName = pdfFile[index1+1:index2]
        image.save(output_dir+ pdfFileName+'.jpg', 'JPEG')
        print('converting ' + str(pdfFile_index +1) + ' th: ' + pdfFileName)

        
   
# convert all images in equation folder

tempfold_dir ='/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataSet/equation_plans/'
os.chdir(tempfold_dir)
tempOut_dir = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataSet/equation_plans/output/'
convertPdf2JPG(tempfold_dir, tempOut_dir)


# convert all images in Eko folder
EKO_fold_dir ='/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataSet/eko_plans/'
os.chdir(EKO_fold_dir)
EKO_Out_dir = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataSet/eko_plans/output/'
convertPdf2JPG(EKO_fold_dir, EKO_Out_dir)

# convert all images in oxygen_plans folder
oxy_fold_dir ='/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataSet/oxygen_plans/'
os.chdir(oxy_fold_dir)
oxy_Out_dir = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataSet/oxygen_plans/output/'
convertPdf2JPG(folder_dir=oxy_fold_dir, output_dir=oxy_Out_dir)