import numpy as np
from manim import *
from src.SlideFrames import *
from src.configs import *
from src.mobjects.faces import HumanHappyFace, HumanSadFace, HumanNeutralFace, BotHappyFace, BotSadFace, BotNeutralFace
from src.hopfield_tools import HopfieldNetworkTools

from SlideComponents.S02_RAMvsCAM.SC01_RAM import RAMSlideComponent
from SlideComponents.S02_RAMvsCAM.SC02_BUT import BUTSlideComponent
from SlideComponents.S02_RAMvsCAM.SC03_CAM import CAMSlideComponent
from SlideComponents.S02_RAMvsCAM.SC04_GangUp import GangingUpSlideComponent
from SlideComponents.S02_RAMvsCAM.SC05_Preset import PreSetSlideComponent
from SlideComponents.S02_RAMvsCAM.SC06_SelfOrg import SelfOrgSlideComponent
from SlideComponents.S02_RAMvsCAM.SC07_MesdUp import MessedUpSlideComponent
from SlideComponents.S02_RAMvsCAM.SC08_SubCons import SubConsSlideComponent

class RAMvsCAM(SlideWithCover):
    def construct(self):
        self.add_cover("存储与记忆：『事实』如何被唤起")

        #self.slide_manager.add_component(RAMSlideComponent)
        #self.slide_manager.add_component(BUTSlideComponent)
        #self.slide_manager.add_component(CAMSlideComponent)
        #self.slide_manager.add_component(GangingUpSlideComponent)
        self.slide_manager.add_component(PreSetSlideComponent)
        #self.slide_manager.add_component(SelfOrgSlideComponent)
        self.slide_manager.add_component(MessedUpSlideComponent)
        #self.slide_manager.add_component(SubConsSlideComponent)
        #self.slide_manager.add_component(ThanksSlideComponent)

        self.slide_manager.simple_execute_all()