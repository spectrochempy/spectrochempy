{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}
     {% if methods %}
    .. rubric:: Methods:

    .. autosummary::

       {% for item in methods %}
       {{ name }}.{{ item }}
       {%- endfor %}

      {% endif %}
    {% endblock %}


    {% block attributes %}
     {% if attributes %}
    .. rubric:: Attributes:

       {% for item in attributes %}
    .. autoattribute:: {{ item }}
       {%- endfor %}

     {% endif %}
    {% endblock %}


.. include:: /gettingstarted/gallery/backreferences/{{fullname}}.examples

.. raw:: html

   <div style='clear:both'></div>
