"""AI-Generated Text Detection System.

Two-stage classical ML cascade (Stage 1: raw AI vs human, Stage 2: humanized AI
vs human) plus always-on supporting evidence (rule-based style checker, GPT-2
perplexity/burstiness). See SRS_AI_Text_Detection.md for the full spec.
"""

from .detector import Detector, DetectionResult

__all__ = ["Detector", "DetectionResult"]

__version__ = "1.0.0"
