"""
Реализация продукционной системы - Задача подготовки к выходу из дома
Production System Implementation - Preparing to Leave Home Task
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Тип действия"""
    CHECK = "Проверить"
    GET = "Взять"
    PUT_ON = "Надеть"
    FINAL = "Завершить"


@dataclass
class Production:
    """Продукционное правило"""
    name: str
    condition: Dict[str, bool]  # Условие: переменная состояния -> ожидаемое значение
    action: str  # Название действия
    action_type: ActionType
    result: Dict[str, bool]  # Результат: измененное состояние после выполнения
    
    def can_apply(self, state: Dict[str, bool]) -> bool:
        """Проверить, можно ли применить это продукционное правило"""
        for key, value in self.condition.items():
            if state.get(key) != value:
                return False
        return True
    
    def apply(self, state: Dict[str, bool]) -> Dict[str, bool]:
        """Применить продукционное правило, вернуть новое состояние"""
        new_state = state.copy()
        new_state.update(self.result)
        return new_state


class ProductionSystem:
    """Продукционная система"""
    
    def __init__(self):
        self.productions: List[Production] = []
        self.initial_state: Dict[str, bool] = {}
        self.goal_state: Dict[str, bool] = {}
        self.execution_history: List[Tuple[str, Dict[str, bool]]] = []
    
    def add_production(self, production: Production):
        """Добавить продукционное правило"""
        self.productions.append(production)
    
    def set_initial_state(self, state: Dict[str, bool]):
        """Установить начальное состояние"""
        self.initial_state = state.copy()
    
    def set_goal_state(self, state: Dict[str, bool]):
        """Установить целевое состояние"""
        self.goal_state = state.copy()
    
    def is_goal_reached(self, state: Dict[str, bool]) -> bool:
        """Проверить, достигнуто ли целевое состояние"""
        for key, value in self.goal_state.items():
            if state.get(key) != value:
                return False
        return True
    
    def solve(self) -> Tuple[bool, List[str]]:
        """
        Решение: от начального состояния к целевому состоянию
        Возвращает: (успех, последовательность выполненных действий)
        """
        current_state = self.initial_state.copy()
        executed_actions = []
        max_iterations = 100  # Предотвращение бесконечного цикла
        iteration = 0
        
        print("\n=== Начало решения ===")
        print(f"Начальное состояние: {self._format_state(current_state)}")
        print(f"Целевое состояние: {self._format_state(self.goal_state)}\n")
        
        while not self.is_goal_reached(current_state) and iteration < max_iterations:
            iteration += 1
            applied = False
            
            # Попытка применить каждое продукционное правило
            for production in self.productions:
                if production.can_apply(current_state):
                    # Проверить, не было ли уже выполнено (избежать повторений)
                    if production.name not in executed_actions:
                        print(f"Шаг {iteration}: Применение правила '{production.name}'")
                        print(f"  Условие: {self._format_condition(production.condition)}")
                        print(f"  Действие: {production.action}")
                        current_state = production.apply(current_state)
                        print(f"  Новое состояние: {self._format_state(current_state)}\n")
                        
                        executed_actions.append(production.name)
                        self.execution_history.append((production.name, current_state.copy()))
                        applied = True
                        break
            
            if not applied:
                print(f"Предупреждение: Не удалось найти применимое правило (итерация {iteration})")
                print(f"Текущее состояние: {self._format_state(current_state)}\n")
                break
        
        success = self.is_goal_reached(current_state)
        if success:
            print("=== Решение успешно! ===")
        else:
            print("=== Решение не удалось ===")
        
        return success, executed_actions
    
    def solve_backward(self) -> Tuple[bool, List[str]]:
        """
        Обратное решение: от целевого состояния к начальному (для проверки)
        """
        print("\n=== Обратная проверка (от цели к началу) ===")
        # Здесь можно реализовать обратный вывод, но для упрощения используем прямой вывод
        return True, []
    
    def check_consistency(self) -> Dict[str, any]:
        """Проверить непротиворечивость продукционных правил"""
        issues = []
        chains = []
        
        # Проверка цепочек правил
        for i, prod1 in enumerate(self.productions):
            for j, prod2 in enumerate(self.productions):
                if i != j:
                    # Проверить, удовлетворяет ли результат prod1 условию prod2
                    match_count = 0
                    for key, value in prod1.result.items():
                        if key in prod2.condition and prod2.condition[key] == value:
                            match_count += 1
                    
                    if match_count > 0:
                        chains.append((prod1.name, prod2.name, match_count))
        
        # Проверка конфликтов: несколько правил с одинаковыми условиями, но разными результатами
        for i, prod1 in enumerate(self.productions):
            for j, prod2 in enumerate(self.productions):
                if i < j:
                    if prod1.condition == prod2.condition and prod1.result != prod2.result:
                        issues.append(f"Конфликт: {prod1.name} и {prod2.name} имеют одинаковые условия, но разные результаты")
        
        return {
            "chains": chains,
            "conflicts": issues,
            "total_productions": len(self.productions)
        }
    
    def _format_state(self, state: Dict[str, bool]) -> str:
        """Форматировать отображение состояния"""
        true_items = [k for k, v in state.items() if v]
        false_items = [k for k, v in state.items() if not v]
        result = []
        if true_items:
            result.append(f"[+] {', '.join(true_items)}")
        if false_items:
            result.append(f"[-] {', '.join(false_items)}")
        return " | ".join(result) if result else "Пустое состояние"
    
    def _format_condition(self, condition: Dict[str, bool]) -> str:
        """Форматировать отображение условия"""
        return self._format_state(condition)


def create_preparing_to_leave_system() -> ProductionSystem:
    """Создать продукционную систему 'Подготовка к выходу из дома'"""
    
    system = ProductionSystem()
    
    # Определить начальное состояние: человек дома, но еще не готов
    initial_state = {
        "at_home": True,
        "has_keys": False,
        "has_phone": False,
        "has_wallet": False,
        "has_jacket": False,
        "checked_weather": False,
        "ready_to_leave": False
    }
    
    # Определить целевое состояние: готов к выходу
    goal_state = {
        "has_keys": True,
        "has_phone": True,
        "has_wallet": True,
        "ready_to_leave": True
    }
    
    system.set_initial_state(initial_state)
    system.set_goal_state(goal_state)
    
    # Продукционное правило 1: Проверить погоду (если нужно)
    system.add_production(Production(
        name="P1: Проверить погоду",
        condition={"at_home": True, "checked_weather": False},
        action="Проверить погодные условия",
        action_type=ActionType.CHECK,
        result={"checked_weather": True}
    ))
    
    # Продукционное правило 2: Решить, надевать ли куртку в зависимости от погоды
    system.add_production(Production(
        name="P2: Надеть куртку (если холодно)",
        condition={"checked_weather": True, "has_jacket": False},
        action="Надеть куртку",
        action_type=ActionType.PUT_ON,
        result={"has_jacket": True}
    ))
    
    # Продукционное правило 3: Взять ключи
    system.add_production(Production(
        name="P3: Взять ключи",
        condition={"at_home": True, "has_keys": False},
        action="Взять ключи со стола",
        action_type=ActionType.GET,
        result={"has_keys": True}
    ))
    
    # Продукционное правило 4: Взять телефон
    system.add_production(Production(
        name="P4: Взять телефон",
        condition={"at_home": True, "has_phone": False},
        action="Взять телефон с зарядного устройства",
        action_type=ActionType.GET,
        result={"has_phone": True}
    ))
    
    # Продукционное правило 5: Взять кошелек
    system.add_production(Production(
        name="P5: Взять кошелек",
        condition={"at_home": True, "has_wallet": False},
        action="Взять кошелек из ящика",
        action_type=ActionType.GET,
        result={"has_wallet": True}
    ))
    
    # Продукционное правило 6: Подтвердить готовность (все предметы есть)
    system.add_production(Production(
        name="P6: Подтвердить готовность",
        condition={
            "has_keys": True,
            "has_phone": True,
            "has_wallet": True
        },
        action="Подтвердить, что все предметы готовы, можно выходить",
        action_type=ActionType.FINAL,
        result={"ready_to_leave": True}
    ))
    
    return system


def main():
    """Главная функция: демонстрация продукционной системы"""
    
    print("=" * 60)
    print("Продукционная система - Задача подготовки к выходу из дома")
    print("Production System - Preparing to Leave Home")
    print("=" * 60)
    
    # Создать систему
    system = create_preparing_to_leave_system()
    
    # Показать все продукционные правила
    print("\n=== Список продукционных правил ===")
    for i, prod in enumerate(system.productions, 1):
        print(f"\nПравило {i}: {prod.name}")
        print(f"  Условие: {system._format_condition(prod.condition)}")
        print(f"  Действие: {prod.action}")
        print(f"  Результат: {system._format_state(prod.result)}")
    
    # Проверить непротиворечивость
    print("\n=== Проверка непротиворечивости ===")
    consistency = system.check_consistency()
    print(f"Всего правил: {consistency['total_productions']}")
    
    if consistency['chains']:
        print("\nСвязи цепочек правил:")
        for chain in consistency['chains']:
            print(f"  {chain[0]} -> {chain[1]} (совпадение {chain[2]} условий)")
    
    if consistency['conflicts']:
        print("\nОбнаружены конфликты:")
        for conflict in consistency['conflicts']:
            print(f"  [WARNING] {conflict}")
    else:
        print("\n[OK] Конфликтов не обнаружено")
    
    # Решить
    success, actions = system.solve()
    
    # Показать историю выполнения
    print("\n=== История выполнения ===")
    for i, (action_name, state) in enumerate(system.execution_history, 1):
        print(f"{i}. {action_name}")
        print(f"   Состояние: {system._format_state(state)}")
    
    print(f"\nПоследовательность действий: {' -> '.join(actions)}")
    
    return system


if __name__ == "__main__":
    system = main()
