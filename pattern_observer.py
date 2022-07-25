
from abc import abstractmethod

class Observable:
    def __init__(self,name):
        self._name = name
        self._observers = []
        
    def attach_observer(self,observer):
        self._observers.append(observer)
        
    def remove_observer(self,observer):
        self._observers.remove(observer)
        
    def notify_changes(self, value):
        for observer in self._observers:
            observer.notify_changes(ObservableData(self._name,value))
            
            
class Observer:
    def __init__(self,name):
        self._name = name
    
    @abstractmethod
    def notify_changes(self, change):
        pass

class ObservableData:
    def __init__(self,name,data):
        self._name = name
        self._data = data
        


"""
class TestObservable(Observable):
    def __init__(self, name):
        Observable.__init__(self,name)
        
class TestObserver(Observer):
    def __init__(self, name):
        Observer.__init__(self,name)
    def notify_changes(self, change):
        print(f" receive changes from {change._name} : {change._data}")
        
        
if __name__ == "__main__":
    test_observer = TestObserver("observer test")
    test_observable = TestObservable("observable test")
    test_observable.attach_observer(test_observer)
    test_observable.notify_changes("notify changes")
"""