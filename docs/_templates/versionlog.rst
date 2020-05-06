.. _version_{{target}}:

Version {{ target }}
---------------------

Bugs fixed
~~~~~~~~~~~
{% for item in bugs.index  %}
{% set fields = bugs.loc[item] -%}
* FIX #{{ item }} - {{ fields.Category }}: {{ fields.Subject }}
{%- endfor %}

Features added
~~~~~~~~~~~~~~~~
{% for item in features.index  %}
{% set fields = features.loc[item] -%}
* #{{ item }} - {{ fields.Category }}: {{ fields.Subject }}
{%- endfor %}

Tasks terminated
~~~~~~~~~~~~~~~~~
{% for item in tasks.index  %}
{% set fields = tasks.loc[item] -%}
* #{{ item }} - {{ fields.Category }}: {{ fields.Subject }}
{%- endfor %}


