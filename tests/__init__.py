# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from hypothesis import settings


settings.register_profile("dqmc", deadline=None, max_examples=500,
                          report_multiple_bugs=True, derandomize=True)
