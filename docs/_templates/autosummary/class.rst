
{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{% block methods %}
{% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
{% for item in methods %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
{% block attributes %}
{% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :nosignatures:
{% for item in attributes %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. include:: /gen_modules/backreferences/{{fullname}}.examples

.. raw:: html

   <div style='clear:both'></div>