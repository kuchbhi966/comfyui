import totoro.sd
import totoro.model_sampling
import totoro.latent_formats
import nodes


class AddFluxFlowNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "flow": ("FLOW",),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, flow):
        m = model.clone()
        model_options = {**model.model_options}
        model.model_options = model_options
        transformer_options = {**model.model_options.get('transformer_options', {})}
        model.model_options['transformer_options'] = transformer_options

        transformer_options['FLOW'] = flow

        return (m, )

