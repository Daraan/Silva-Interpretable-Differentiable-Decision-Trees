from ray.rllib.core.models.catalog import Catalog

import functools


class DDTCatalog(Catalog):
    def _determine_components_hook(self) -> None:
        """Hook to determine the components of the model."""
        # We do not need an encoder; no not set _encoder_config, hence do not call super()

        assert not hasattr(self, "_encoder_config")

        # Create a function that can be called when framework is known to retrieve the
        # class type for action distributions
        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space,
            action_space=self.action_space,
        )
