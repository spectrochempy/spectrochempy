{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {%- if item != "__init__" %}
      ~{{ name }}.{{ item }}
    {%- endif%}
   {%- endfor %}

   {% for item in methods %}
   {%- if item != "__init__" %}
   .. automethod:: {{ item }}
   {%- endif%}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. include:: /gettingstarted/gallery/backreferences/{{fullname}}.examples

.. raw:: html

   <div style='clear:both'></div>
