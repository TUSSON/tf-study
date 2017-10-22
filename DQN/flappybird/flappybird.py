#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QHBoxLayout,
                             QLabel, QApplication)
from PyQt5.QtGui import QPixmap, QFont, QPalette
from PyQt5.QtCore import QThread, QBasicTimer, QRect, Qt
from resource import loadResource
import numpy as np
import random
import sys

class ImageItem(QLabel):
    def __init__(self, window, pixmap=None, hitmask=None):
        super().__init__(window)
        if type(pixmap) is not list:
            pixmap = [pixmap] 
            hitmask = [hitmask]
        self.index = 0
        self.setPixmap(pixmap[0])
        self.pixmap = pixmap
        self.hitmask = hitmask
        self.window = window

    def nextPixmap(self):
        self.index = (self.index + 1) % len(self.pixmap)
        self.setPixmap(self.pixmap[self.index])

    def checkOverlap(self, imgItem):
        ar = self.geometry()
        br = imgItem.geometry()
        if not ar.intersects(br):
            return False

        am = self.hitmask[self.index].copy()
        bm = imgItem.hitmask[imgItem.index].copy()
        am += ar.y() * 1000 + ar.x()
        bm += br.y() * 1000 + br.x()
         
        count = np.intersect1d(am, bm).size
        return count > 0

class PipeItem:
    def __init__(self, window, pixmap, hitmask=None):
        self.pipel = ImageItem(window, pixmap[0], hitmask[0])
        self.pipeu = ImageItem(window, pixmap[1], hitmask[1])

    def movepos(self, pos):
        self.pos = pos
        self.pipeu.move(pos['x'], pos['yu'])
        self.pipel.move(pos['x'], pos['yl'])

    def move(self):
        self.pos['x'] -= 4
        self.movepos(self.pos)

    def getx(self):
        return self.pos['x']
        
    def checkOverlap(self, bird):
        up = self.pipel.checkOverlap(bird)
        lo = self.pipeu.checkOverlap(bird)
        return up or lo

class BirdItem(ImageItem):
    def __init__(self, window, pixmap, hitmask=None):
        super().__init__(window, pixmap, hitmask)
        self.speed = 0
        self.maxspeed = 10
        self.minspeed = -8
        self.maxy = int(window.basey - pixmap[0].size().height())
        self.acc = 1
        self.loopIndex = 0
        self.flapacc = -9

    def update(self, action):
        self.loopIndex = (self.loopIndex + 1) % 30

        if self.loopIndex % 3 == 0:
            self.nextPixmap()

        if self.speed < self.maxspeed and not action:
            self.speed += self.acc
        elif self.speed > self.minspeed and action:
            self.speed = self.flapacc
        
        y = int(self.y())
        y += self.speed
        if y < 0:
            self.speed = 0
            y = 0
        elif y > self.maxy:
            self.speed = 0
            y = self.maxy

        self.move(self.x(), y)

class FlappyBirdUI(QWidget):
    def __init__(self):
        super().__init__()
        self.pixmaps, self.hitmask = loadResource()
        self.initUI()

    def initUI(self):
        self.screensize = self.pixmaps['bg'].size()
        self.pipesize = self.pixmaps['pipe'][0].size()
        self.basey = self.screensize.height()*0.79
        self.scorey = self.screensize.height()*0.89

        self.setWindowTitle('Flappy Bird')
        self.setFixedSize(self.screensize)
        self.timer = QBasicTimer()

        ImageItem(self, self.pixmaps['bg'])

        self.pipes = []
        for i in range(3):
            pipe = PipeItem(self, self.pixmaps['pipe'], self.hitmask['pipe'])
            self.pipes.append(pipe)

        self.bird = BirdItem(self, self.pixmaps['bird'], self.hitmask['bird'])

        self.base = ImageItem(self, self.pixmaps['base'])
        self.baseshift = self.pixmaps['base'].size().width() - self.screensize.width()

        self.scoreLabel = QLabel(self)
        self.scoreLabel.setFixedWidth(200)
        self.scoreLabel.setAlignment(Qt.AlignCenter)
        self.scoreLabel.setFont(QFont("", 16, QFont.Bold))
        self.scoreLabel.move(self.screensize.width()/2-100, self.scorey)

        self.restart()
        self.show()
        self.timer.start(30, self)

    def restart(self):
        pipegapx = self.screensize.width() / 2
        i = 0
        for pipe in self.pipes:
            pipepos = self.getRandomPipePos()
            pipepos['x'] += pipegapx * i
            i+=1
            pipe.movepos(pipepos)

        posx = self.screensize.width()*0.2
        posy = self.screensize.height()/2
        self.bird.move(posx, posy)

        self.basex = 0
        self.base.move(self.basex, self.basey)

        self.score = 0
        self.scoreLabel.setNum(0)
        self.action = 0
        pass


    def getRandomPipePos(self):
        gapY = random.randint(2,9)*10
        gapY += self.screensize.height() * 0.16
        pipeX = 1.5*self.screensize.width() - self.pipesize.width()

        return {'x': pipeX, 'yu': gapY - self.pipesize.height(), 'yl': gapY + 100}

    def isCrash(self):
        for pip in self.pipes:
            crash = pip.checkOverlap(self.bird)
            if crash:
                return True

        if self.bird.y() >= self.bird.maxy:
            return True

        return False

    def timerEvent(self, event):
        for pipe in self.pipes:
            pipe.move()
            if pipe.getx() < - self.pipesize.width():
                    pipe.movepos(self.getRandomPipePos())

            birdmx = self.bird.x() + self.bird.width()/2
            pipemx = pipe.getx() + self.pipesize.width()/2
            if pipemx <= birdmx < pipemx + 4:
                self.score+=1
                self.scoreLabel.setNum(self.score)
                self.scoreLabel.repaint()

        self.bird.update(self.action)
        self.action = 0

        if self.isCrash():
            self.restart()
            
        self.basex = -((-self.basex + 4) % self.baseshift)
        self.base.move(self.basex, self.basey)

    def mousePressEvent(self, event):
       self.action = 1    

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = FlappyBirdUI()
    sys.exit(app.exec_())
