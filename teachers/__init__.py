"""Specialist teachers."""
from typing import Dict

from teachers.base import BaseTeacher, TeacherResponse
from teachers.code_teacher import CodeTeacher
from teachers.general_teacher import GeneralTeacher
from teachers.math_teacher import MathTeacher
from teachers.science_teacher import ScienceTeacher

__all__ = [
    "BaseTeacher",
    "TeacherResponse",
    "MathTeacher",
    "CodeTeacher",
    "ScienceTeacher",
    "GeneralTeacher",
    "build_teachers",
]


def build_teachers(client=None) -> Dict[str, BaseTeacher]:
    """name -> teacher instance. Pass one shared client (or leave None to create it lazily)."""
    return {
        "math": MathTeacher(client=client),
        "code": CodeTeacher(client=client),
        "science": ScienceTeacher(client=client),
        "general": GeneralTeacher(client=client),
    }
