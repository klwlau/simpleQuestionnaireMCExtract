import numpy as np
import csv, json, os, ntpath, glob, datetime,time
from joblib import Parallel, delayed
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim


class extractor:

    def __init__(self, configJsonPath):
        print("Read Config Json")
        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 70
        self.configDict = json.load(open(configJsonPath))
        self.dataFolder = self.configDict["folder"]
        self.questionDict = self.configDict["question"]
        self.resultFolderPath = self.makeDirInDataFolder("Result")
        self.fileList = glob.glob(self.dataFolder + "/*.*")
        self.fileList.sort()
        self.refImage = self.readImgFile(self.configDict["refForm"])
        print("Previewing File")
        plt.imshow(self.refImage, cmap="gray")
        plt.show()
        self.diffToRefPath = self.makeDirInDataFolder("diffToRef")
        self.createLabelRefDict()
        self.timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

    def readImgFile(self, imageFilePath):
        # data = np.array(Image.open(imageFilePath))
        data = cv2.imread(imageFilePath, 0)
        return data

    def saveToCSV(self, writeArray, fileName):
        """save a list of row to CSV file"""
        with open(fileName, 'a', newline='') as f:
            csvWriter = csv.writer(f)
            for row in writeArray:
                csvWriter.writerow(row)
        print("save to:" + fileName)

    def makeDirInDataFolder(self, dirName):
        '''make a new directory with dirName if it does not exists'''
        if not os.path.exists(os.path.join(self.dataFolder, dirName)):
            os.makedirs(os.path.join(self.dataFolder, dirName))
            print("make ", dirName, " Dir")
        return os.path.join(self.dataFolder, dirName) + "/"

    def darwBoxWithText(self, imageArray, coordinate, size, boxText="", drawOption=False):
        # print(type((coordinate[0] + size[0], coordinate[1] + size[1])))
        if drawOption:
            imgWithBox = cv2.rectangle(imageArray, coordinate, (coordinate[0] + size[0], coordinate[1] + size[1]),
                                       (0, 0, 255), 3)
            imgWithBoxText = cv2.putText(imgWithBox, boxText, (coordinate[0], coordinate[1] + 30),
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # imgWithBox = cv2.rectangle(imageArray, coordinate, (coordinate[0] + size[0], coordinate[1] + size[1]),
            #                            (0, 255, 0), 3)
            imgWithBoxText = cv2.putText(imageArray, boxText, (coordinate[0] - 50, coordinate[1]),
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
        return imgWithBoxText

    def createLabelRefDict(self):
        labelRefFolderPath = self.makeDirInDataFolder("labelRef")
        self.refLabelDict = {}
        for questionLabel, metaData in self.questionDict.items():
            optionDict = {}
            # optionDict["choice"] = metaData["choice"]
            for optionNumber in range(metaData["choice"]):
                optionMetaData = {}
                box_x_len = round(metaData["size"][0] / metaData["choice"])
                box_y_len = metaData["size"][1]

                x1 = metaData["xy"][0] + optionNumber * box_x_len
                x2 = metaData["xy"][0] + (optionNumber + 1) * box_x_len
                y1 = metaData["xy"][1]
                y2 = metaData["xy"][1] + box_y_len

                optionMetaData["imageArray"] = self.refImage[y1:y2, x1:x2]
                optionMetaData["x1"] = x1
                optionMetaData["x2"] = x2
                optionMetaData["y1"] = y1
                optionMetaData["y2"] = y2

                optionDict[str(optionNumber)] = optionMetaData

                savedFilePath = labelRefFolderPath + questionLabel + "_" + str(optionNumber) + "_labeled.png"

                cv2.imwrite(savedFilePath, optionMetaData["imageArray"])

            self.refLabelDict[str(questionLabel)] = optionDict

    def labelQuestionnaire(self, filePath):

        def drawQuestionnaireBoxes(image, indResultList):
            for counter, (questionLabel, metaData) in enumerate(self.questionDict.items()):
                image = self.darwBoxWithText(image, tuple(metaData["xy"]), metaData["size"], boxText=questionLabel)
                for i in range(metaData["choice"]):
                    box_x_len = round(metaData["size"][0] / metaData["choice"])
                    box_y_len = metaData["size"][1]
                    org_x = metaData["xy"][0] + box_x_len * i
                    org_y = metaData["xy"][1]
                    image = self.darwBoxWithText(image, (org_x, org_y), (box_x_len, box_y_len),
                                                 boxText="option_" + str(i),
                                                 drawOption=True)
                    if i == individualResult[counter]:
                        image = cv2.putText(image, "X", (int(org_x + box_x_len / 4), int(org_y + box_y_len)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

            return image

        def getAnswer(image, filePath):

            questionAnswerList = []
            for questionLabel, questionData in self.refLabelDict.items():
                optionScoreList = []
                plt.figure()
                figCounter = 0

                for optionNumber, optionDict in questionData.items():
                    selectedPart = image[optionDict["y1"]:optionDict["y2"], optionDict["x1"]:optionDict["x2"]]
                    refPart = optionDict["imageArray"]
                    score, diff = compare_ssim(selectedPart, refPart, full=True)
                    optionScoreList.append(score)

                    plt.subplot2grid((3, len(questionData)), (0, figCounter))
                    # plt.title("%.4f" % score)
                    plt.text(2, 2, "%.4f" % score, bbox={'facecolor': 'white', 'pad': 10})
                    plt.imshow(refPart)

                    plt.subplot2grid((3, len(questionData)), (1, figCounter))
                    plt.imshow(selectedPart)
                    plt.subplot2grid((3, len(questionData)), (2, figCounter))
                    plt.imshow(diff, vmin=-1, vmax=1)
                    figCounter += 1

                plt.savefig(
                    self.diffToRefPath + ntpath.basename(filePath)[:-4] + "_" + str(questionLabel) + "_labeled.png")
                plt.clf()
                plt.close('all')

                questionAnswerList.append(optionScoreList.index(min(optionScoreList)))

            return questionAnswerList

        print("labeling:", filePath)
        image = self.readImgFile(filePath)

        image, h = self.alignImages(image, self.refImage)

        individualResult = getAnswer(image, filePath)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        drawQuestionnaireBoxes(image, individualResult)

        savedFilePath = self.resultFolderPath + ntpath.basename(filePath)[:-4] + "_labeled.png"
        cv2.imwrite(savedFilePath, image)

        return [ntpath.basename(filePath)[:-4]]+individualResult

    def alignImages(self, img, imReference):

        im1Gray = img
        im2Gray = imReference


        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)


        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)


        matches.sort(key=lambda x: x.distance, reverse=False)


        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        #label matches
        # imMatches = cv2.drawMatches(img, keypoints1, imReference, keypoints2, matches, None)
        # cv2.imwrite("matches.jpg", imMatches)

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt


        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        height, width = imReference.shape
        im1Reg = cv2.warpPerspective(img, h, (width, height))

        return im1Reg, h

    def main(self):

        print("Start Labeling")

        parrllelResult = Parallel(n_jobs=-1)(delayed(self.labelQuestionnaire)(filePath) for filePath in self.fileList)


        csvFilePath = self.resultFolderPath + "result_"+self.timeStamp+".csv"
        csvHeader = ["FileName"]
        for key,_ in self.configDict["question"].items():
            csvHeader.append(key)

        # print(csvHeader)
        print("Start Saving")
        self.saveToCSV([csvHeader],csvFilePath)
        self.saveToCSV(parrllelResult, csvFilePath)
        print("Done")

