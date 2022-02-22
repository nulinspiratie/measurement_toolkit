

__all__ = ['DotDict', 'UpdateDotDict']

class DotDict(dict):
    """
    Wrapper dict that allows to get dotted attributes
    """
    exclude_from_dict = []
    def __init__(self, value=None):
        if value is None:
            pass
        else:
            for key in value:
                self.__setitem__(key, value[key])

    def __setitem__(self, key, value):
        # string type must be checked, as key could be other datatype
        if type(key)==str and '.' in key:
            myKey, restOfKey = key.split('.', 1)
            target = self.setdefault(myKey, DotDict())
            target[restOfKey] = value
        else:
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
            dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        if type(key) != str or '.' not in key:
            return dict.__getitem__(self, key)
        myKey, restOfKey = key.split('.', 1)
        target = dict.__getitem__(self, myKey)
        return target[restOfKey]

    def __contains__(self, key):
        if not isinstance(key, str) or '.' not in key:
            return dict.__contains__(self, key)
        myKey, restOfKey = key.split('.', 1)

        if myKey not in self:
            return False
        else:
            target = dict.__getitem__(self, myKey)
            return restOfKey in target

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self)))

    # dot acces baby
    def __setattr__(self, key, val):
        if key in self.exclude_from_dict:
            self.__dict__[key] = val
        else:
            self.__setitem__(key, val)

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(f'Attribute {key} not found')

    def __dir__(self):
        # Add keys to dir, used for auto-completion
        items = super().__dir__()
        items.extend(self.keys())
        return items

    def setdefault(self, key, default=None):
        """Set value of a key if it does not yet exist"""
        d = self
        if isinstance(key, str):
            *parent_keys, key = key.split('.')
            for subkey in parent_keys:
                d = dict.setdefault(d, subkey, DotDict())

        return dict.setdefault(d, key, default)

    def create_dicts(self, *keys):
        """Create nested dict structure
        Args:
            *keys: Sequence of key strings. Empty DotDicts will be created if
                each key does not yet exist
        Returns:
            Most inner dict, newly created if it does not yet exist
        Examples:
            d = DotDict()
            d.create_dicts('a', 'b', 'c')
            print(d.a.b.c)
            >>> {}
        """
        d = self
        for key in keys:
            if key in self:
                assert isinstance(d[key], dict)

            d.setdefault(key, DotDict())
            d = d[key]
        return d

class UpdateDotDict(DotDict):
    """DotDict that can evaluate function upon being updated.

    Args:
        update_function: Function that is called every time a value changes.
        **kwargs: Unused kwargs.
    """
    exclude_from_dict = ['update_function', 'exclude_from_dict']

    def __init__(self, update_function=None, **kwargs):
        self.update_function = update_function
        super().__init__()

        for key, val in kwargs.items():
            DotDict.__setitem__(self, key, val)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.update_function is not None:
            self.update_function()
