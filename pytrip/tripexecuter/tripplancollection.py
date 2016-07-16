"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""
from pytrip.tripexecuter.pytripobj import pytripObj


class TripPlanCollection(pytripObj):
    def __init__(self):
        self.save_fields = ["plans"]
        self.plans = []
        self.active_plan = None

    def get_plans(self):
        return self.plans

    def add_plan(self, plan):
        plan.Init(self)
        if plan.name == "":
            plan.set_name("plan %d" % (len(self.plans) + 1))
        self.plans.append(plan)
        self.active_plan = plan

    def __iter__(self):
        self.iter = 0
        return self

    def next(self):
        if self.iter >= len(self.plans):
            raise StopIteration
        self.iter += 1
        return self.plans[self.iter - 1]

    def get_plan_by_id(self, id):
        return self.plans[id]

    def get_plan(self, name):
        for plan in self.plans:
            if plan.name == name:
                return plan
        return None

    def get_active_plan(self):
        return self.active_plan

    def set_active_plan(self, plan):
        self.active_plan = plan

    def delete_plan(self, plan):
        self.plans.remove(plan)

        if self.active_plan is plan:
            new_plan = None
            if len(self.plans) is not 0:
                new_plan = self.plans[0]
            return new_plan

        return None

    def delete_plan_by_id(self, id):
        plan = self.plans.pop(id)
        plan.destroy()
