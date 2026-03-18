import numpy as np
from pathlib import Path
from manim import *
from manim.scene.scene_file_writer import SceneFileWriter
from src.SlideFrames import *
from src.configs import *
from src.mobjects.faces import HumanHappyFace, HumanSadFace, HumanNeutralFace, BotHappyFace, BotSadFace, BotNeutralFace
from src.hopfield_tools import HopfieldNetworkTools

# Safety net: if any partial-movie entry references a file that was never
# rendered (e.g. a component error was swallowed mid-scene), PyAV's concat
# demuxer crashes with a misleading FileNotFoundError on Windows.
# This patch filters the file list to only include files that actually exist.
_original_combine_to_movie = SceneFileWriter.combine_to_movie

def _patched_combine_to_movie(self):
    before = len(self.partial_movie_files)
    self.partial_movie_files = [
        p for p in self.partial_movie_files
        if p is not None and Path(p).exists()
    ]
    after = len(self.partial_movie_files)
    if before != after:
        print(f"[patch] Filtered {before - after} phantom partial movie entries ({before} → {after})")
    _original_combine_to_movie(self)

SceneFileWriter.combine_to_movie = _patched_combine_to_movie

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

        self.slide_manager.add_component(RAMSlideComponent)
        self.slide_manager.add_component(BUTSlideComponent)
        self.slide_manager.add_component(CAMSlideComponent)
        self.slide_manager.add_component(GangingUpSlideComponent)
        self.slide_manager.add_component(PreSetSlideComponent)
        self.slide_manager.add_component(SelfOrgSlideComponent)
        self.slide_manager.add_component(MessedUpSlideComponent)
        self.slide_manager.add_component(SubConsSlideComponent)
        self.slide_manager.add_component(ThanksSlideComponent)

        self.slide_manager.simple_execute_all()