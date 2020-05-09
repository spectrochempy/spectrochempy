:orphan:

Version {{ target }}
---------------------

Bugs fixed
~~~~~~~~~~~
{% for item in bugs.index  %}
{% set fields = bugs.loc[item] -%}
* FIX `#{{ fields['#'] }} <https://redmine.spectrochempy.fr/issues/{{ fields['#'] }}>`_ - {{ fields.Category }}: {{ fields.Subject }}
{%- endfor %}

Features added
~~~~~~~~~~~~~~~~
{% for item in features.index  %}
{% set fields = features.loc[item] -%}
* `#{{ fields['#'] }} <https://redmine.spectrochempy.fr/issues/{{ fields['#'] }}>`_ - {{ fields.Category }}: {{ fields.Subject }}
{%- endfor %}

Tasks terminated
~~~~~~~~~~~~~~~~~
{% for item in tasks.index  %}
{% set fields = tasks.loc[item] -%}
* `#{{ fields['#'] }} <https://redmine.spectrochempy.fr/issues/{{ fields['#'] }}>`_ - {{ fields.Category }}: {{ fields.Subject }}
{%- endfor %}


